"""Stage A (single-axis) verification. Verification checklist from the project spec (section 6)."""
import numpy as np
import pytest

from cubesat.params import DEG, EnvParams, Scenario, SimParams, SpacecraftParams, replace
from cubesat.simulate import run_single_axis, scenario_metrics
from cubesat.params import SLEW, HOLD

SC = SpacecraftParams()


def _second_order_step(t, wn, zeta):
    wd = wn * np.sqrt(1 - zeta ** 2)
    return 1 - np.exp(-zeta * wn * t) * (np.cos(wd * t) + zeta / np.sqrt(1 - zeta ** 2) * np.sin(wd * t))


def test_matches_analytic_second_order():
    """Small gains, no saturation, no noise: J s^2 + Kd s + Kp closed loop."""
    J = SC.J[0]
    wn, zeta = 0.05, 0.7
    Kp, Kd = J * wn ** 2, 2 * zeta * wn * J
    sc = Scenario("small", 0.5, (1, 0, 0), 200.0, "slew")
    r = run_single_axis((Kp, 0.0, Kd), sc, noise=False)
    ref = 0.5 * DEG
    y = (r.err + ref) / ref
    y_th = _second_order_step(r.t, wn, zeta)
    assert np.max(np.abs(r.u_cmd)) < 0.01 * SC.tau_max  # truly unsaturated
    assert np.max(np.abs(y - y_th)) < 0.01  # within 1% of step size


def test_total_momentum_conserved_without_disturbance():
    """Wheel torque is internal: J*omega + h is constant when tau_d = 0 and no clamp."""
    J = SC.J[0]
    wn, zeta = 0.3, 0.8
    r = run_single_axis((J * wn ** 2, 0.0, 2 * zeta * wn * J), SLEW, noise=False)
    assert not r.h_clamped.any()
    L = J * r.rate + r.h
    assert np.max(np.abs(L - L[0])) < 1e-12


def test_saturation_limits_respected():
    """Aggressive gains: commanded torque clamped, wheel momentum never above h_max."""
    r = run_single_axis((1.0, 0.01, 0.5), SLEW, seed=1)
    assert np.max(np.abs(r.u_cmd)) <= SC.tau_max + 1e-15
    assert np.max(np.abs(r.h)) <= SC.h_max + 1e-15


def test_momentum_clamp_engages_under_large_disturbance():
    env = replace(EnvParams(), dist_bias=(5e-5, 0, 0), dist_sin_amp=(0, 0, 0))
    J = SC.J[0]
    r = run_single_axis((J * 0.3 ** 2, 1e-4, 2 * 0.8 * 0.3 * J), HOLD, env=env, seed=0)
    # 5e-5 N m * 600 s = 0.03 N m s > h_max: wheel must hit the limit
    assert r.h_clamped.any()
    assert np.max(np.abs(r.h)) <= SC.h_max + 1e-15
    m = scenario_metrics(r, HOLD, SC)
    assert m["h_violation"] and m["h_peak_frac"] == pytest.approx(1.0)


def test_step_size_convergence():
    J = SC.J[0]
    g = (J * 0.3 ** 2, 1e-5, 2 * 0.8 * 0.3 * J)
    m1 = scenario_metrics(run_single_axis(g, SLEW, sim=SimParams(dt=0.01), seed=3), SLEW, SC)
    m2 = scenario_metrics(run_single_axis(g, SLEW, sim=SimParams(dt=0.005), seed=3), SLEW, SC)
    assert abs(m1["t_settle"] - m2["t_settle"]) <= 0.02  # within 2 integration steps
    assert abs(m1["overshoot"] - m2["overshoot"]) < 0.01  # percentage points
    assert abs(m1["h_peak_frac"] - m2["h_peak_frac"]) < 1e-3


def test_determinism():
    g = (5e-3, 1e-5, 2e-2)
    a = run_single_axis(g, HOLD, seed=7)
    b = run_single_axis(g, HOLD, seed=7)
    c = run_single_axis(g, HOLD, seed=8)
    assert np.array_equal(a.err, b.err)
    assert not np.array_equal(a.err, c.err)
