"""Robust constrained BO: the same BO as optim/bo.py, applied to a worst-case
problem over K fixed training plants (sampled with TRAIN_SEED, disjoint from
the robustness test plants).

Each evaluation simulates the design on all K plants and reports, for every
metric, the WORST value across plants (max t_settle, max overshoot, max hold
RMS, max momentum and saturation fractions; diverged if any plant diverges).
A design is feasible only if it meets every spec on every training plant.

Cost note (reported honestly): one robust evaluation = K simulator calls, so
at the same number of optimizer evaluations robust BO uses K times more
simulation. The comparison with nominal BO is about the formulation, not
sample efficiency.
"""
from dataclasses import dataclass

import numpy as np

from cubesat.problem import Problem
from cubesat.uncertainty import TRAIN_SEED, sample_plants

from . import bo

K_PLANTS = 8
MAX_KEYS = ["t_settle", "overshoot", "e_hold_rms_deg", "hold_t_settle", "h_peak_frac",
            "sat_frac", "tau_peak_frac"]


@dataclass
class RobustProblem(Problem):
    k_plants: int = K_PLANTS

    def __post_init__(self):
        super().__post_init__()
        self.plants = sample_plants(self.k_plants, seed=TRAIN_SEED, sc0=self.sc, env0=self.env)

    def simulate(self, gains, sc=None, env=None, slew_seed=None, hold_seed=None):
        if sc is not None or env is not None:   # explicit plant -> ordinary evaluation
            return super().simulate(gains, sc, env, slew_seed, hold_seed)
        ms = [super(RobustProblem, self).simulate(gains, s.sc, s.env, s.slew_seed, s.hold_seed)[0]
              for s in self.plants]
        agg = {k: float(max(m[k] for m in ms)) for k in MAX_KEYS}
        agg["energy"] = float(np.mean([m["energy"] for m in ms]))
        agg["h_violation"] = any(m["h_violation"] for m in ms)
        agg["diverged"] = any(m["diverged"] for m in ms)
        agg["n_plants_feasible"] = int(sum(self.feasible(m) for m in ms))
        return agg, None


def optimize(problem, budget, seed):
    rp = RobustProblem(stage=problem.stage, specs=problem.specs, sc=problem.sc, env=problem.env,
                       sim=problem.sim)
    return bo.optimize(rp, budget, seed)
