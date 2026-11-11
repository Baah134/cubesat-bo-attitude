"""Genetic algorithm baseline (pymoo GA, default operators) with pymoo's native
feasibility-first constraint handling (G <= 0 feasible), same budget as others."""
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem as PymooProblem
from pymoo.core.termination import NoTermination

from .common import BudgetTracker

POP = 10


def optimize(problem, budget, seed):
    f = BudgetTracker(problem, budget)
    pp = PymooProblem(n_var=problem.dim, n_obj=1, n_ieq_constr=problem.n_constraints,
                      xl=0.0, xu=1.0)
    algo = GA(pop_size=POP, eliminate_duplicates=True)
    algo.setup(pp, termination=NoTermination(), seed=seed, verbose=False)
    while not f.done():
        pop = algo.ask()
        X = pop.get("X")[: f.remaining]
        recs = [f(x) for x in X]
        if len(X) < len(pop):
            break
        F = np.array([[r["outcomes"][0]] for r in recs])
        G = np.array([r["outcomes"][1:] for r in recs])
        pop.set("F", F, "G", G)
        Evaluator().eval(pp, pop, skip_already_evaluated=True)
        algo.tell(infills=pop)
    return f.history
