import numpy as np

from cubesat.problem import Problem
from cubesat.uncertainty import TEST_SEED, TRAIN_SEED, sample_plants
from optim.common import BudgetTracker, best_feasible, first_feasible_iter


def test_gain_mapping_roundtrip():
    p = Problem("B")
    x = np.random.default_rng(0).uniform(0, 1, p.dim)
    assert np.allclose(p.to_unit(p.to_gains(x)), x)


def test_outcomes_finite_for_diverging_or_extreme_designs():
    p = Problem("A")
    for x in ([1, 1, 0], [0, 0, 0], [1, 0, 1], [1, 1, 1]):
        r = p.evaluate(x)
        assert np.all(np.isfinite(r["outcomes"]))
        assert np.all(np.isfinite(r["constraints"]))


def test_best_feasible_never_returns_infeasible():
    h = [{"feasible": False, "objective": 1.0, "iter": 0},
         {"feasible": True, "objective": 5.0, "iter": 1},
         {"feasible": False, "objective": 0.1, "iter": 2}]
    assert best_feasible(h)["objective"] == 5.0
    assert first_feasible_iter(h) == 2
    assert best_feasible([h[0]]) is None


def test_budget_tracker_enforces_budget():
    p = Problem("A")
    f = BudgetTracker(p, 2)
    f([0.5] * 3)
    f([0.5] * 3)
    try:
        f([0.5] * 3)
        assert False, "should raise"
    except RuntimeError:
        pass


def test_plant_samples_deterministic_disjoint_and_valid():
    a, b = sample_plants(20, TEST_SEED), sample_plants(20, TEST_SEED)
    c = sample_plants(20, TRAIN_SEED)
    assert [s.factors for s in a] == [s.factors for s in b]
    assert a[0].factors != c[0].factors
    for s in a:
        assert np.all(np.linalg.eigvalsh(s.sc.J_matrix) > 0)
        assert all(0.7 <= f <= 1.3 for f in s.factors["J_factor"])
        assert 0.5 <= s.factors["dist_factor"] <= 3.0
