"""The IECON 2026 method: Optuna TPE with a constraints_func (c <= 0 feasible).

Reuses the formulation of Baah-Mensah & Armah, IECON 2026
(github.com/Baah134/iecon-2026-pid-bo-tuning, src/bo_tuning.py):
objective = settling time; constraints returned through trial user_attrs;
infeasible/unstable designs get violated constraints. Differences, recorded in
docs/DECISIONS.md: parameters are suggested in the same normalized log-gain
unit cube used by every other method, the startup (random) phase is 2d+1
trials to match the BO initial design, and the budget is equal to the others.
"""
import optuna

from .common import BudgetTracker

optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize(problem, budget, seed):
    f = BudgetTracker(problem, budget)

    def objective(trial):
        x = [trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(problem.dim)]
        rec = f(x)
        trial.set_user_attr("constraints", tuple(rec["constraints"]))
        return rec["objective"]

    sampler = optuna.samplers.TPESampler(
        seed=seed, constraints_func=lambda t: t.user_attrs.get("constraints", (1.0,) * problem.n_constraints),
        n_startup_trials=2 * problem.dim + 1)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=budget)
    return f.history
