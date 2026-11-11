"""Constraint-aware Bayesian optimization (BoTorch), the main new method.

- Inputs: normalized log-gain unit cube [0,1]^d.
- One independent GP per outcome (ModelListGP of SingleTaskGP, Matern-5/2 ARD,
  BoTorch defaults), outputs standardized. Outcomes from Problem.outcomes():
  [log t_settle, c_overshoot, c_hold, c_h, c_sat] with c <= 0 feasible
  (heavy-tailed metrics in log-ratio form).
- Acquisition: qLogNoisyExpectedImprovement with BoTorch `constraints=`
  callables (sigmoid feasibility weighting), q = 1. Until the first feasible
  point exists, qLogProbabilityOfFeasibility is maximized instead (BoTorch's
  recommended handling; see docs/DECISIONS.md).
- Initial design: scrambled Sobol, 2d+1 points.
"""
import warnings

import numpy as np
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement, qLogProbabilityOfFeasibility
from botorch.acquisition.objective import GenericMCObjective
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.mlls import SumMarginalLogLikelihood
from scipy.stats import qmc

from .common import BudgetTracker

torch.set_default_dtype(torch.float64)


def _fit(X, Y):
    models = []
    for j in range(Y.shape[1]):
        models.append(SingleTaskGP(X, Y[:, j:j + 1], outcome_transform=Standardize(m=1)))
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit_gpytorch_mll(mll)
        except Exception:  # fall back to default hyperparameters; recorded in history
            pass
    return model


def optimize(problem, budget, seed, num_restarts=10, raw_samples=256, mc_samples=128):
    torch.manual_seed(seed)
    f = BudgetTracker(problem, budget)
    d = problem.dim
    n_init = min(2 * d + 1, budget)
    for x in qmc.Sobol(d, scramble=True, seed=seed).random(n_init):
        f(x)
    bounds = torch.stack([torch.zeros(d), torch.ones(d)])
    nc = problem.n_constraints
    objective = GenericMCObjective(lambda Z, X=None: -Z[..., 0])
    constraints = [(lambda Z, j=j: Z[..., 1 + j]) for j in range(nc)]
    while not f.done():
        X = torch.tensor([r["x"] for r in f.history])
        Y = torch.tensor([r["outcomes"] for r in f.history])
        model = _fit(X, Y)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]), seed=seed)
        if any(r["feasible"] for r in f.history):
            acq = qLogNoisyExpectedImprovement(
                model=model, X_baseline=X, objective=objective, constraints=constraints,
                sampler=sampler, prune_baseline=True)
        else:
            # no feasible point yet: maximize (log) probability of feasibility
            acq = qLogProbabilityOfFeasibility(model=model, constraints=constraints,
                                               sampler=sampler)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cand, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=num_restarts,
                                    raw_samples=raw_samples,
                                    options={"seed": seed + len(f.history)})
        f(cand.detach().numpy().ravel())
    return f.history
