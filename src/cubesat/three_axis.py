"""Stage B: three-axis rigid body with three orthogonal reaction wheels.

Quaternion convention: SCALAR FIRST, q = [q0, q1, q2, q3], Hamilton product,
q maps body -> inertial (v_I = q (x) v_B (x) q*). Body rates omega in rad/s.

State vector s (length 14):
    s[0:4]   q
    s[4:7]   omega            body rate
    s[7:10]  h                wheel momenta (wheel spin relative to body), body axes
    s[10:13] tau              torque the motors apply to the body (first-order lag)
    s[13]    E                mechanical work at the wheels, int sum|tau_i * omega_w_i| dt

Dynamics:
    q_dot     = 0.5 * q (x) [0, omega]
    J omega_dot = -omega x (J omega + h) + tau_eff + tau_d
    h_dot     = -tau_eff
    tau_dot   = (u_sat - tau) / T_m
Momentum limit per wheel as in Stage A. Disturbances are applied in body axes
(simplification; documented in docs/EXPLAINER.md).
Total inertial angular momentum R(q) (J omega + h) is conserved when tau_d = 0
and no momentum clamp is active; tests/test_stage_b.py checks this.
"""
import numpy as np

from ._jit import njit
from .controller import pid_axis

# Log-space gain bounds. x/y axes share gains (equal inertia); z has ~6x less inertia.
BOUNDS_B = {
    "Kp_xy": (1e-5, 1.0), "Ki_xy": (1e-8, 1e-2), "Kd_xy": (1e-4, 1.0),
    "Kp_z": (1e-6, 1.0), "Ki_z": (1e-9, 1e-2), "Kd_z": (1e-5, 1.0),
}


@njit
def qmul(a0, a1, a2, a3, b0, b1, b2, b3):
    return (a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
            a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
            a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0)


@njit
def _deriv(s, u, td, J, Jinv, J_w, T_m, h_max, out):
    q0, q1, q2, q3 = s[0], s[1], s[2], s[3]
    wx, wy, wz = s[4], s[5], s[6]
    te = np.empty(3)
    for i in range(3):
        t = s[10 + i]
        h = s[7 + i]
        if (h >= h_max and -t > 0.0) or (h <= -h_max and -t < 0.0):
            t = 0.0
        te[i] = t
    Hx = J[0, 0] * wx + J[0, 1] * wy + J[0, 2] * wz + s[7]
    Hy = J[1, 0] * wx + J[1, 1] * wy + J[1, 2] * wz + s[8]
    Hz = J[2, 0] * wx + J[2, 1] * wy + J[2, 2] * wz + s[9]
    # -omega x H
    mx = -(wy * Hz - wz * Hy) + te[0] + td[0]
    my = -(wz * Hx - wx * Hz) + te[1] + td[1]
    mz = -(wx * Hy - wy * Hx) + te[2] + td[2]
    out[0] = -0.5 * (q1 * wx + q2 * wy + q3 * wz)
    out[1] = 0.5 * (q0 * wx + q2 * wz - q3 * wy)
    out[2] = 0.5 * (q0 * wy + q3 * wx - q1 * wz)
    out[3] = 0.5 * (q0 * wz + q1 * wy - q2 * wx)
    out[4] = Jinv[0, 0] * mx + Jinv[0, 1] * my + Jinv[0, 2] * mz
    out[5] = Jinv[1, 0] * mx + Jinv[1, 1] * my + Jinv[1, 2] * mz
    out[6] = Jinv[2, 0] * mx + Jinv[2, 1] * my + Jinv[2, 2] * mz
    E = 0.0
    for i in range(3):
        out[7 + i] = -te[i]
        out[10 + i] = (u[i] - s[10 + i]) / T_m
        E += abs(te[i] * s[7 + i])
    out[13] = E / J_w


@njit
def _disturbance(t, bias, amp, period, phase, drand, out):
    for i in range(3):
        out[i] = bias[i] + amp[i] * np.sin(2.0 * np.pi * t / period + phase[i]) + drand[i]


@njit
def error_vector(q, qr):
    """Attitude error e = 2 sign(qe0) qe_vec with qe = qr* (x) q (shortest rotation)."""
    e0, e1, e2, e3 = qmul(qr[0], -qr[1], -qr[2], -qr[3], q[0], q[1], q[2], q[3])
    sg = 1.0 if e0 >= 0.0 else -1.0
    return e0 * sg, 2.0 * sg * e1, 2.0 * sg * e2, 2.0 * sg * e3


@njit
def _log_state(s, k, q_ref, qi, axis, t, t_log, ang_log, trav_log, w_log, h_log, tau_log, q_log):
    e0, ex, ey, ez = error_vector(s[0:4], q_ref)
    t_log[k] = t
    ang_log[k] = 2.0 * np.arctan2(0.5 * np.sqrt(ex * ex + ey * ey + ez * ez), e0)
    r0, r1, r2, r3 = qmul(qi[0], qi[1], qi[2], qi[3], s[0], s[1], s[2], s[3])
    nv = np.sqrt(r1 * r1 + r2 * r2 + r3 * r3)
    if nv > 0.0:
        trav_log[k] = 2.0 * np.arctan2(nv, r0) / nv * (r1 * axis[0] + r2 * axis[1] + r3 * axis[2])
    else:
        trav_log[k] = 0.0
    for i in range(3):
        w_log[k, i] = s[4 + i]
        h_log[k, i] = s[7 + i]
        tau_log[k, i] = s[10 + i]
    for i in range(4):
        q_log[k, i] = s[i]


@njit
def simulate_kernel(q_init, w_init, q_ref, gains, i_clamp, J, Jinv, J_w, T_m, tau_max, h_max,
                    bias, amp, period, phase, drand, att_noise, gyro_noise, gyro_bias,
                    axis, horizon, dt, ctrl_dt, controller_on):
    n_steps = int(round(horizon / dt))
    sub = int(round(ctrl_dt / dt))
    n_ctrl = n_steps // sub
    t_log = np.empty(n_steps + 1)
    ang_log = np.empty(n_steps + 1)     # principal error angle to q_ref (rad, >= 0)
    trav_log = np.empty(n_steps + 1)    # rotation from start, projected on slew axis (rad)
    w_log = np.empty((n_steps + 1, 3))
    h_log = np.empty((n_steps + 1, 3))
    tau_log = np.empty((n_steps + 1, 3))
    q_log = np.empty((n_steps + 1, 4))
    u_log = np.zeros((n_ctrl, 3))
    sat_log = np.zeros(n_ctrl, dtype=np.bool_)
    hcl_log = np.zeros(n_steps + 1, dtype=np.bool_)

    s = np.zeros(14)
    s[0:4] = q_init
    s[4:7] = w_init
    k1 = np.empty(14)
    k2 = np.empty(14)
    k3 = np.empty(14)
    k4 = np.empty(14)
    tmp = np.empty(14)
    u = np.zeros(3)
    I = np.zeros(3)
    em = np.zeros(3)
    qm = np.empty(4)
    td1 = np.empty(3)
    td2 = np.empty(3)
    td4 = np.empty(3)
    qi = np.array([q_init[0], -q_init[1], -q_init[2], -q_init[3]])

    t = 0.0
    k = 0
    diverged = False
    _log_state(s, 0, q_ref, qi, axis, t, t_log, ang_log, trav_log, w_log, h_log, tau_log, q_log)
    for c in range(n_ctrl):
        # ----- controller sample (ZOH) -----
        if controller_on:
            n1, n2, n3 = 0.5 * att_noise[c, 0], 0.5 * att_noise[c, 1], 0.5 * att_noise[c, 2]
            nn = np.sqrt(1.0 + n1 * n1 + n2 * n2 + n3 * n3)
            qm[0], qm[1], qm[2], qm[3] = qmul(s[0], s[1], s[2], s[3], 1.0 / nn, n1 / nn, n2 / nn, n3 / nn)
            _, em[0], em[1], em[2] = error_vector(qm, q_ref)
            anysat = False
            for i in range(3):
                j = 0 if i < 2 else 3
                rate = s[4 + i] + gyro_bias[i] + gyro_noise[c, i]
                ui, Ii, si = pid_axis(em[i], rate, I[i], gains[j], gains[j + 1], gains[j + 2],
                                      ctrl_dt, i_clamp, tau_max)
                u[i] = ui
                I[i] = Ii
                anysat = anysat or si
                u_log[c, i] = ui
            sat_log[c] = anysat
        dr = drand[c]
        for st in range(sub):
            _disturbance(t, bias, amp, period, phase, dr, td1)
            _disturbance(t + 0.5 * dt, bias, amp, period, phase, dr, td2)
            _disturbance(t + dt, bias, amp, period, phase, dr, td4)
            _deriv(s, u, td1, J, Jinv, J_w, T_m, h_max, k1)
            for i in range(14):
                tmp[i] = s[i] + 0.5 * dt * k1[i]
            _deriv(tmp, u, td2, J, Jinv, J_w, T_m, h_max, k2)
            for i in range(14):
                tmp[i] = s[i] + 0.5 * dt * k2[i]
            _deriv(tmp, u, td2, J, Jinv, J_w, T_m, h_max, k3)
            for i in range(14):
                tmp[i] = s[i] + dt * k3[i]
            _deriv(tmp, u, td4, J, Jinv, J_w, T_m, h_max, k4)
            for i in range(14):
                s[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
            nq = np.sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2] + s[3] * s[3])
            for i in range(4):
                s[i] /= nq
            clamped = False
            for i in range(3):
                if s[7 + i] > h_max:
                    s[7 + i] = h_max
                elif s[7 + i] < -h_max:
                    s[7 + i] = -h_max
                if abs(s[7 + i]) >= h_max:
                    clamped = True
            t += dt
            k += 1
            _log_state(s, k, q_ref, qi, axis, t, t_log, ang_log, trav_log, w_log, h_log, tau_log, q_log)
            hcl_log[k] = clamped
            wn = abs(s[4]) + abs(s[5]) + abs(s[6])
            if not np.isfinite(wn) or wn > 3.0:
                diverged = True
                break
        if diverged:
            break
    return (t_log[:k + 1], ang_log[:k + 1], trav_log[:k + 1], w_log[:k + 1], h_log[:k + 1],
            tau_log[:k + 1], q_log[:k + 1], u_log, sat_log, hcl_log[:k + 1], s[13], diverged)


def axis_angle_quat(axis, angle):
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * a])


def run_three_axis(gains, scenario, sc, env, sim, seed=0, i_clamp_frac=0.25, noise=True,
                   w_init=(0.0, 0.0, 0.0), controller_on=True):
    """Stage B run. gains = (Kp_xy, Ki_xy, Kd_xy, Kp_z, Ki_z, Kd_z)."""
    from .params import DEG
    from .simulate import RunResult, _noise
    n_ctrl = int(round(scenario.horizon / sim.ctrl_dt))
    att, gyro, drand, phase = _noise(seed, n_ctrl, 3, env)
    ds = env.dist_scale
    gb = np.array(env.gyro_bias, dtype=float)
    if not noise:
        att[:] = 0.0
        gyro[:] = 0.0
        drand[:] = 0.0
        ds = 0.0
        gb[:] = 0.0
    ang = scenario.angle_deg * DEG
    axis = np.asarray(scenario.axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    if scenario.kind == "slew":
        q0, qr = np.array([1.0, 0, 0, 0]), axis_angle_quat(axis, ang)
    else:
        q0, qr = axis_angle_quat(axis, ang), np.array([1.0, 0, 0, 0])
    J = sc.J_matrix
    out = simulate_kernel(q0, np.asarray(w_init, dtype=float), qr, np.asarray(gains, dtype=float),
                          i_clamp_frac * sc.tau_max, J, np.linalg.inv(J), sc.J_wheel, sc.T_m,
                          sc.tau_max, sc.h_max,
                          np.array(env.dist_bias) * ds, np.array(env.dist_sin_amp) * ds,
                          env.orbit_period, phase, drand, att, gyro, gb, axis,
                          scenario.horizon, sim.dt, sim.ctrl_dt, controller_on)
    t, angl, trav, w, h, tau, q, u, sat, hcl, E, div = out
    r = RunResult(t, angl, w, h, tau, u, sat, hcl, float(E), bool(div), scenario.horizon, travel=trav)
    r.q = q
    return r
