"""Shared optimizer plumbing.

Every optimizer implements  optimize(problem, budget, seed) -> list[record]
where each record is the dict returned by Problem.evaluate(x) plus an
"iter" index. Budgets count simulator evaluations exactly: methods that work
in populations are truncated to the budget.
"""
import numpy as np


class BudgetTracker:
    def __init__(self, problem, budget):
        self.problem = problem
        self.budget = budget
        self.history = []

    @property
    def remaining(self):
        return self.budget - len(self.history)

    def done(self):
        return self.remaining <= 0

    def __call__(self, x):
        if self.done():
            raise RuntimeError("evaluation budget exhausted")
        rec = self.problem.evaluate(np.clip(np.asarray(x, dtype=float), 0.0, 1.0))
        rec["iter"] = len(self.history)
        self.history.append(rec)
        return rec


def best_feasible(history):
    """Lowest objective among feasible evaluations, or None (never an infeasible point)."""
    feas = [r for r in history if r["feasible"]]
    return min(feas, key=lambda r: r["objective"]) if feas else None


def best_so_far_curve(history, horizon):
    """Best feasible objective after each evaluation (nan until first feasible)."""
    out, best = [], np.nan
    for r in history:
        if r["feasible"] and (np.isnan(best) or r["objective"] < best):
            best = r["objective"]
        out.append(best)
    return np.array(out)


def first_feasible_iter(history):
    for r in history:
        if r["feasible"]:
            return r["iter"] + 1  # number of evaluations used
    return None
