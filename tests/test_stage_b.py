"""Stage B (three-axis) verification. See CLAUDE.md section 6."""
import numpy as np
import pytest

from cubesat.params import DEG, HOLD, SLEW, EnvParams, Scenario, SimParams, SpacecraftParams, replace
from cubesat.simulate import run_single_axis, scenario_metrics
from cubesat.three_axis import axis_angle_quat, error_vector, qmul, run_three_axis

SC = SpacecraftParams()
ENV = EnvParams()
SIM = SimParams()
G_NOM = (0.042 * 0.3 ** 2, 1e-5, 2 * 0.8 * 0.3 * 0.042, 0.0067 * 0.3 ** 2, 1e-6, 2 * 0.8 * 0.3 * 0.0067)


def _rot(q):
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)]])


def _inertial_H(r, J):
    return np.array([_rot(q) @ (J @ w + h) for q, w, h in zip(r.q, r.rate, r.h)])


def test_torque_free_rigid_body_conserves_energy_and_momentum():
    sc = replace(SC, J_offdiag=(0.002, -0.001, 0.0005))
    J = sc.J_matrix
    sc_ = Scenario("free", 0.0, (1, 0, 0), 100.0, "hold")
    r = run_three_axis((0,) * 6, sc_, sc, ENV, SIM, noise=False, w_init=(0.01, 0.02, 0.03),
                       controller_on=False)
    T = 0.5 * np.einsum("ni,ij,nj->n", r.rate, J, r.rate)
    H = _inertial_H(r, J)
    assert np.max(np.abs(T - T[0])) / T[0] < 1e-9
    assert np.max(np.linalg.norm(H - H[0], axis=1)) / np.linalg.norm(H[0]) < 1e-9
    # the motion is genuinely 3-D (rates change), so this is a non-trivial check
    assert np.ptp(r.rate[:, 0]) > 1e-4


def test_internal_wheel_torque_conserves_inertial_momentum():
    r = run_three_axis(G_NOM, SLEW, SC, ENV, SIM, noise=False)
    assert not r.h_clamped.any()
    H = _inertial_H(r, SC.J_matrix)
    assert np.max(np.linalg.norm(H - H[0], axis=1)) < 1e-12


def test_quaternion_norm_long_run():
    r = run_three_axis(G_NOM, HOLD, SC, ENV, SIM, seed=0)
    assert np.max(np.abs(np.linalg.norm(r.q, axis=1) - 1.0)) < 1e-9


def test_error_quaternion_no_unwinding():
    """A 350 deg rotation about x is a -10 deg error, not +350."""
    q = axis_angle_quat((1, 0, 0), 350 * DEG)
    e0, ex, ey, ez = error_vector(q, np.array([1.0, 0, 0, 0]))
    assert e0 > 0
    assert ex == pytest.approx(2 * np.sin(-5 * DEG), rel=1e-12)
    assert abs(ey) < 1e-15 and abs(ez) < 1e-15


def test_qmul_identity_and_inverse():
    q = axis_angle_quat((1, 2, 3), 1.0)
    assert np.allclose(qmul(*q, 1.0, 0, 0, 0), q)
    assert np.allclose(qmul(*q, q[0], -q[1], -q[2], -q[3]), (1, 0, 0, 0))


def test_reduces_to_stage_a_for_small_single_axis_slew():
    """Small slew about body x, no noise: three-axis model == single-axis model."""
    g = G_NOM[:3]
    sc_ = Scenario("small", 0.5, (1, 0, 0), 120.0, "slew")
    ra = run_single_axis(g, sc_, noise=False)
    rb = run_three_axis((*g, *G_NOM[3:]), sc_, SC, ENV, SIM, noise=False)
    # small-angle: 2 sin(phi/2) vs phi differs by ~phi^3/24 -> negligible at 0.5 deg
    assert np.max(np.abs(rb.travel - (ra.err + 0.5 * DEG))) < 1e-6 * DEG * 100
    assert np.max(np.abs(rb.rate[:, 1:])) < 1e-15


def test_limits_respected_aggressive_gains():
    r = run_three_axis((1.0, 0.01, 0.5, 1.0, 0.01, 0.5), SLEW, SC, ENV, SIM, seed=2)
    assert np.max(np.abs(r.u_cmd)) <= SC.tau_max + 1e-15
    assert np.max(np.abs(r.h)) <= SC.h_max + 1e-15


def test_momentum_clamp_under_large_disturbance():
    env = replace(ENV, dist_bias=(5e-5, 0, 0), dist_sin_amp=(0, 0, 0))
    r = run_three_axis(G_NOM, HOLD, SC, env, SIM, seed=0)
    assert r.h_clamped.any()
    assert np.max(np.abs(r.h)) <= SC.h_max + 1e-15


def test_slew_reaches_target_and_metrics_sane():
    # PD (Ki = 0): G_NOM's very slow integrator (Ti ~ 380 s) winds up during the slew
    # and leaves a slow tail, which is real controller behavior, not what this test checks.
    g = (G_NOM[0], 0.0, G_NOM[2], G_NOM[3], 0.0, G_NOM[5])
    r = run_three_axis(g, SLEW, SC, ENV, SIM, seed=0)
    m = scenario_metrics(r, SLEW, SC)
    assert not m["diverged"]
    assert m["t_settle"] < SLEW.horizon
    assert abs(r.travel[-1] - 30 * DEG) < 0.1 * DEG


def test_step_size_convergence():
    m1 = scenario_metrics(run_three_axis(G_NOM, SLEW, SC, ENV, SimParams(dt=0.01), seed=3), SLEW, SC)
    m2 = scenario_metrics(run_three_axis(G_NOM, SLEW, SC, ENV, SimParams(dt=0.005), seed=3), SLEW, SC)
    assert abs(m1["t_settle"] - m2["t_settle"]) <= 0.02
    assert abs(m1["overshoot"] - m2["overshoot"]) < 0.01
    assert abs(m1["h_peak_frac"] - m2["h_peak_frac"]) < 1e-3


def test_determinism():
    a = run_three_axis(G_NOM, HOLD, SC, ENV, SIM, seed=5)
    b = run_three_axis(G_NOM, HOLD, SC, ENV, SIM, seed=5)
    assert np.array_equal(a.err, b.err) and np.array_equal(a.q, b.q)
