"""M1 spec calibration: evaluate a Sobol sweep of the gain space (Stage A or B)
and report the distribution of each metric, so the hold-RMS spec can be set
"achievable but not trivial". Output: results/calibration_<stage>.json

Usage: python -m scripts.calibrate_specs --stage A --n 512
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from cubesat.problem import Problem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="A")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()
    p = Problem(stage=a.stage)
    X = qmc.Sobol(p.dim, scramble=True, seed=a.seed).random(a.n)
    t0 = time.perf_counter()
    rows = [p.evaluate(x) for x in X]
    wall = time.perf_counter() - t0
    M = {k: np.array([r["metrics"][k] for r in rows], dtype=float)
         for k in ["t_settle", "overshoot", "e_hold_rms_deg", "h_peak_frac", "sat_frac", "diverged"]}
    ok = M["diverged"] == 0
    settled = ok & (M["t_settle"] < 120.0)
    q = [5, 10, 25, 50, 75, 90]
    summ = {
        "stage": a.stage, "n": a.n, "seed": a.seed, "wall_s": wall, "s_per_eval": wall / a.n,
        "frac_diverged": float(1 - ok.mean()),
        "frac_slew_settled": float(settled.mean()),
        "percentiles": q,
        "settled_designs": {k: np.percentile(v[settled], q).tolist() for k, v in M.items() if k != "diverged"}
        if settled.any() else {},
        "frac_feasible_current_specs": float(np.mean([r["feasible"] for r in rows])),
        "e_hold_rms_deg_all_ok": np.percentile(M["e_hold_rms_deg"][ok], q).tolist(),
    }
    # How the feasible fraction depends on the hold spec, other specs fixed
    others = (M["overshoot"] <= p.specs.max_overshoot) & (M["h_peak_frac"] <= p.specs.max_h_peak_frac) \
        & (M["sat_frac"] <= p.specs.max_sat_frac) & ok
    summ["feasible_frac_vs_hold_spec"] = {
        str(s): float(np.mean(others & (M["e_hold_rms_deg"] <= s)))
        for s in [0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.05]}
    summ["frac_meeting_other_specs"] = float(others.mean())
    Path("results").mkdir(exist_ok=True)
    Path(f"results/calibration_{a.stage}.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
