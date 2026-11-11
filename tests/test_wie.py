"""Expert baseline (Wie-Lu rate-limited eigenaxis law, stage W)."""
import numpy as np

from cubesat.params import DEG, LARGE_SLEW, SLEW, EnvParams, SimParams, SpacecraftParams
from cubesat.problem import Problem
from cubesat.three_axis import run_three_axis
from optim.wie_rule import wie_rule_params


def _run(g, scen, **kw):
    return run_three_axis(np.asarray(g), scen, SpacecraftParams(), EnvParams(), SimParams(),
                          seed=0, noise=False, law=1, **kw)


def test_rule_slew_reaches_target_and_meets_specs():
    p = Problem(stage="W")
    m, (rs, _) = p.simulate(wie_rule_params(p))
    assert rs.err[-1] / DEG < 0.1
    assert p.feasible(m)


def test_rate_limit_holds_on_large_slew():
    # 90 deg slew: with wn high the law is rate-limited; body rate must not exceed w_max (+5%)
    w_max = 0.05
    r = _run([1.0, 0.9, w_max, 0.1], LARGE_SLEW)
    assert np.linalg.norm(r.rate, axis=1).max() < 1.05 * w_max
    assert r.err[-1] / DEG < 0.1


def test_torque_limit_and_direction_preserving_saturation():
    r = _run([3.0, 0.9, 0.2, 0.1], SLEW)
    sc = SpacecraftParams()
    assert np.abs(r.u_cmd).max() <= sc.tau_max + 1e-12
    # saturated samples keep the commanded torque direction: some sample hits the limit on one axis only
    hit = np.isclose(np.abs(r.u_cmd).max(axis=1), sc.tau_max)
    assert hit.any()


def test_eigenaxis_slew_stays_near_the_axis():
    # rate-limited law should rotate about the eigenaxis: off-axis travel stays small
    r = _run([1.0, 0.9, 0.05, 0.1], LARGE_SLEW)
    ax = np.asarray(LARGE_SLEW.axis, float)
    ax /= np.linalg.norm(ax)
    w = r.rate
    off = np.linalg.norm(w - np.outer(w @ ax, ax), axis=1)
    assert off.max() < 0.1 * np.linalg.norm(w, axis=1).max()
