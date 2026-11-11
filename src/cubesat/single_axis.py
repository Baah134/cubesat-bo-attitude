"""Stage A: single-axis rigid body + one reaction wheel (axis x).

States (integrated with fixed-step RK4):
    theta   body angle about x (rad)
    omega   body rate (rad/s)
    h       wheel momentum (N m s), h = J_w * omega_w (wheel speed relative to body)
    tau     torque the motor currently applies to the body (N m), first-order lag
    E       accumulated mechanical work at the wheel, int |tau * omega_w| dt (J)

Dynamics:
    J * theta_ddot = tau_eff + tau_d          (tau_eff: torque actually applied to body)
    h_dot          = -tau_eff                 (equal and opposite on the wheel)
    tau_dot        = (u_cmd_sat - tau) / T_m
Momentum limit: if |h| >= h_max and the torque would increase |h|, tau_eff = 0
(the wheel cannot spin faster). This is recorded as a momentum violation.
"""
import numpy as np

from ._jit import njit
from .controller import pid_axis


@njit
def _deriv(theta, omega, h, tau, u_cmd, tau_d, J, J_w, T_m, h_max):
    tau_eff = tau
    if (h >= h_max and -tau > 0.0) or (h <= -h_max and -tau < 0.0):
        tau_eff = 0.0
    d_theta = omega
    d_omega = (tau_eff + tau_d) / J
    d_h = -tau_eff
    d_tau = (u_cmd - tau) / T_m
    d_E = abs(tau_eff * h / J_w)
    return d_theta, d_omega, d_h, d_tau, d_E


@njit
def simulate_kernel(theta0, ref, Kp, Ki, Kd, i_clamp,
                    J, J_w, T_m, tau_max, h_max,
                    dist_bias, dist_amp, orbit_period, dist_phase, dist_rand,
                    att_noise, gyro_noise, gyro_bias,
                    horizon, dt, ctrl_dt):
    n_steps = int(round(horizon / dt))
    sub = int(round(ctrl_dt / dt))
    n_ctrl = n_steps // sub
    t_log = np.empty(n_steps + 1)
    th_log = np.empty(n_steps + 1)
    om_log = np.empty(n_steps + 1)
    h_log = np.empty(n_steps + 1)
    tau_log = np.empty(n_steps + 1)
    u_log = np.empty(n_ctrl)
    sat_log = np.zeros(n_ctrl, dtype=np.bool_)
    hclamp_log = np.zeros(n_steps + 1, dtype=np.bool_)

    theta = theta0
    omega = 0.0
    h = 0.0
    tau = 0.0
    E = 0.0
    I = 0.0
    t = 0.0
    diverged = False
    t_log[0] = 0.0
    th_log[0] = theta
    om_log[0] = omega
    h_log[0] = h
    tau_log[0] = tau
    k = 0
    two_pi = 2.0 * np.pi
    for c in range(n_ctrl):
        # --- controller sample (ZOH) ---
        e_meas = (theta + att_noise[c]) - ref
        r_meas = omega + gyro_bias + gyro_noise[c]
        u, I, sat = pid_axis(e_meas, r_meas, I, Kp, Ki, Kd, ctrl_dt, i_clamp, tau_max)
        u_log[c] = u
        sat_log[c] = sat
        d_rand = dist_rand[c]
        for s in range(sub):
            td1 = dist_bias + dist_amp * np.sin(two_pi * t / orbit_period + dist_phase) + d_rand
            tmid = t + 0.5 * dt
            td2 = dist_bias + dist_amp * np.sin(two_pi * tmid / orbit_period + dist_phase) + d_rand
            td4 = dist_bias + dist_amp * np.sin(two_pi * (t + dt) / orbit_period + dist_phase) + d_rand
            a1, b1, c1, e1, f1 = _deriv(theta, omega, h, tau, u, td1, J, J_w, T_m, h_max)
            a2, b2, c2, e2, f2 = _deriv(theta + 0.5 * dt * a1, omega + 0.5 * dt * b1,
                                        h + 0.5 * dt * c1, tau + 0.5 * dt * e1, u, td2,
                                        J, J_w, T_m, h_max)
            a3, b3, c3, e3, f3 = _deriv(theta + 0.5 * dt * a2, omega + 0.5 * dt * b2,
                                        h + 0.5 * dt * c2, tau + 0.5 * dt * e2, u, td2,
                                        J, J_w, T_m, h_max)
            a4, b4, c4, e4, f4 = _deriv(theta + dt * a3, omega + dt * b3,
                                        h + dt * c3, tau + dt * e3, u, td4,
                                        J, J_w, T_m, h_max)
            theta += dt / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
            omega += dt / 6.0 * (b1 + 2 * b2 + 2 * b3 + b4)
            h += dt / 6.0 * (c1 + 2 * c2 + 2 * c3 + c4)
            tau += dt / 6.0 * (e1 + 2 * e2 + 2 * e3 + e4)
            E += dt / 6.0 * (f1 + 2 * f2 + 2 * f3 + f4)
            clamped = False
            if h > h_max:
                h = h_max
                clamped = True
            elif h < -h_max:
                h = -h_max
                clamped = True
            t += dt
            k += 1
            t_log[k] = t
            th_log[k] = theta
            om_log[k] = omega
            h_log[k] = h
            tau_log[k] = tau
            hclamp_log[k] = clamped or abs(h) >= h_max
            if not (np.isfinite(theta) and np.isfinite(omega)) or abs(omega) > 1.0:
                diverged = True
                break
        if diverged:
            break
    return (t_log[:k + 1], th_log[:k + 1], om_log[:k + 1], h_log[:k + 1], tau_log[:k + 1],
            u_log, sat_log, hclamp_log[:k + 1], E, diverged)
