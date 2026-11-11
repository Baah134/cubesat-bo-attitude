"""CMA-ES baseline (pycma) on the unit cube with a feasibility-first penalty.

Constraint handling: Problem.penalized() ranks every feasible design above
every infeasible one, and infeasible designs by total (log-ratio) violation.
This mirrors the penalty approach used in the IECON comparison, but with a
budget equal to every other method's.
"""
import cma
import numpy as np

from .common import BudgetTracker


def optimize(problem, budget, seed):
    f = BudgetTracker(problem, budget)
    opts = cma.CMAOptions()
    opts.set({"bounds": [0.0, 1.0], "seed": seed + 1, "verbose": -9, "maxfevals": budget})
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.25, 0.75, problem.dim)
    es = cma.CMAEvolutionStrategy(x0, 0.3, opts)
    while not f.done():
        X = es.ask()
        X = X[: f.remaining]
        vals = [problem.penalized(f(x)["metrics"]) for x in X]
        if len(X) < es.popsize:
            break  # budget exhausted mid-generation; partial generation is not told
        es.tell(X, vals)
        if es.stop() and not f.done():
            # restart from a new random point if CMA-ES converges early (keeps budget equal)
            es = cma.CMAEvolutionStrategy(rng.uniform(0.0, 1.0, problem.dim), 0.3, opts)
    return f.history
