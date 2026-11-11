"""Random search baseline: scrambled Sobol points in the unit cube (log-gain space)."""
from scipy.stats import qmc

from .common import BudgetTracker


def optimize(problem, budget, seed):
    f = BudgetTracker(problem, budget)
    X = qmc.Sobol(problem.dim, scramble=True, seed=seed).random(budget)
    for x in X:
        f(x)
    return f.history
