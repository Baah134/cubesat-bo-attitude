"""Constraint-aware Bayesian optimization (BoTorch), the main new method.

- Inputs: normalized log-gain unit cube [0,1]^d.
- One independent GP per outcome (ModelListGP of SingleTaskGP, BoTorch default
  Matern-5/2 ARD kernel and priors), outputs standardized. Outcomes from
  Problem.outcomes(): [log t_settle, c_overshoot, c_hold, c_h, c_sat], c <= 0
  feasible (heavy-tailed metrics in log-ratio form).
- Acquisition: analytic LogConstrainedExpectedImprovement (log of EI times the
  product of the GP probabilities that each constraint is <= 0), minimizing
  log t_settle, with best_f = best feasible observed value. The nominal
  evaluation is deterministic (fixed noise seeds), so EI against the observed
  incumbent is appropriate. Until the first feasible point exists, the analytic
  LogProbabilityOfFeasibility is maximized instead.
  (The MC qLogNEI variant was tried first; its acquisition optimization took
  ~57 s per iteration vs ~1 s here, see docs/DECISIONS.md.)
- Initial design: scrambled Sobol, 2d+1 points.
"""
import warnings

import numpy as np
import torch
from botorch.acquisition.analytic import (LogConstrainedExpectedImprovement,
                                          LogProbabilityOfFeasibility)
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import SumMarginalLogLikelihood
from scipy.stats import qmc

from .common import BudgetTracker

torch.set_default_dtype(torch.float64)


def _fit(X, Y):
    models = [SingleTaskGP(X, Y[:, j:j + 1], outcome_transform=Standardize(m=1))
              for j in range(Y.shape[1])]
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit_gpytorch_mll(mll)
        except Exception:  # keep default hyperparameters if fitting fails
            pass
    return model


def optimize(problem, budget, seed, num_restarts=10, raw_samples=512):
    torch.manual_seed(seed)
    f = BudgetTracker(problem, budget)
    d = problem.dim
    n_init = min(2 * d + 1, budget)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X0 = qmc.Sobol(d, scramble=True, seed=seed).random(n_init)
    for x in X0:
        f(x)
    bounds = torch.stack([torch.zeros(d), torch.ones(d)])
    cons = {1 + j: (None, 0.0) for j in range(problem.n_constraints)}
    while not f.done():
        X = torch.tensor([r["x"] for r in f.history])
        Y = torch.tensor([r["outcomes"] for r in f.history])
        model = _fit(X, Y)
        feas = [r["outcomes"][0] for r in f.history if r["feasible"]]
        if feas:
            acq = LogConstrainedExpectedImprovement(model, best_f=min(feas), objective_index=0,
                                                    constraints=cons, maximize=False)
        else:
            acq = LogProbabilityOfFeasibility(model, cons)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cand, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=num_restarts,
                                    raw_samples=raw_samples,
                                    options={"seed": seed * 1000 + len(f.history)})
        f(cand.detach().numpy().ravel())
    return f.history
