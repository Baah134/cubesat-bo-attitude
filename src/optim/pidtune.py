"""MATLAB pidtune baseline (zero optimization budget).

Gains come from scripts/pidtune_baseline.m (MATLAB R2024b, Control System
Toolbox), stored in configs/pidtune_gains.json, so this module does not need
MATLAB at run time. Stage A uses the x/y-axis design.
"""
import json
from pathlib import Path

import numpy as np

GAINS_FILE = Path(__file__).resolve().parents[2] / "configs" / "pidtune_gains.json"


def pidtune_gains(problem):
    d = json.loads(GAINS_FILE.read_text())
    xy, z = d["xy"], d["z"]
    if problem.stage == "A":
        return np.array([xy["Kp"], xy["Ki"], xy["Kd"]])
    return np.array([xy["Kp"], xy["Ki"], xy["Kd"], z["Kp"], z["Ki"], z["Kd"]])


def optimize(problem, budget=0, seed=0):
    g = pidtune_gains(problem)
    lo, hi = 10 ** problem.log_lo, 10 ** problem.log_hi
    if np.any(g < lo) or np.any(g > hi):
        raise ValueError(f"pidtune gains {g} outside search bounds; widen bounds")
    rec = problem.evaluate(problem.to_unit(g))
    rec["iter"] = 0
    return [rec]
