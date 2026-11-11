"""Monte Carlo robustness of each method's best design (per seed) over N sampled plants.

Usage: python -m scripts.robustness --stage B --n 200 [--workers 10]
Writes results/robustness_stage<S>.json (per-design, per-plant metrics) and
results/tables/robustness_stage<S>.{csv,md}. Stage A uses the x-axis inertia
sample (single-axis model ignores products of inertia).
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.style import METHOD_ORDER, label

FIXED = {"pidtune", "analytic", "wie_rule"}


def _eval_design(args):
    stage, gains, n, specs = args
    import warnings
    warnings.filterwarnings("ignore")
    from cubesat.problem import Problem, Specs
    from cubesat.uncertainty import sample_plants
    p = Problem(stage=stage, specs=Specs(**specs))
    rows = []
    for s in sample_plants(n):
        m, _ = p.simulate(np.asarray(gains), sc=s.sc, env=s.env, slew_seed=s.slew_seed,
                          hold_seed=s.hold_seed)
        rows.append({"plant": s.index, **m, "compliant": p.feasible(m)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="B")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra history dirs, e.g. results/histories/stageB_robust")
    a = ap.parse_args()
    summ = json.loads(Path(f"results/stage{a.stage}_summary.json").read_text())
    specs = json.loads(next(Path(f"results/histories/stage{a.stage}").glob("*.json")).read_text())["specs"]
    designs = []
    for m in [x for x in METHOD_ORDER if x in summ["methods"]]:
        d = summ["methods"][m]
        for seed, g in zip(d["seeds"], d["best_gains_per_seed"]):
            if g is not None:
                designs.append((m, seed, g))
    for ex in a.extra:
        for p in sorted(Path(ex).glob("*.json")):
            r = json.loads(p.read_text())
            from optim.common import best_feasible
            b = best_feasible(r["history"])
            if b is not None:
                designs.append((r["method"], r["seed"], b["gains"]))
    jobs = [(a.stage, g, a.n, specs) for _, _, g in designs]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(_eval_design, jobs))
    out = {"stage": a.stage, "n_plants": a.n, "plant_seed": 777, "specs": specs, "designs": []}
    for (m, seed, g), rows in zip(designs, res):
        out["designs"].append({"method": m, "seed": seed, "gains": g, "plants": rows})
    Path(f"results/robustness_stage{a.stage}.json").write_text(json.dumps(out), encoding="utf-8")

    # per-method table: pool all designs of a method (per-seed compliance, then mean +- std)
    tab = []
    methods = list(dict.fromkeys(d["method"] for d in out["designs"]))
    for m in methods:
        ds = [d for d in out["designs"] if d["method"] == m]
        comp = [np.mean([r["compliant"] for r in d["plants"]]) * 100 for d in ds]
        ts = np.array([r["t_settle"] for d in ds for r in d["plants"]])
        hold = np.array([r["e_hold_rms_deg"] for d in ds for r in d["plants"]])
        hv = np.array([r["h_violation"] for d in ds for r in d["plants"]])
        tab.append({
            "method": label(a.stage, m),
            "designs": len(ds),
            "spec compliance % (mean ± std over designs)": f"{np.mean(comp):.1f} ± {np.std(comp, ddof=1) if len(comp) > 1 else 0:.1f}",
            "min compliance %": f"{np.min(comp):.1f}",
            "t_settle median [s]": f"{np.median(ts):.2f}",
            "t_settle 95th pct [s]": f"{np.percentile(ts, 95):.2f}",
            "worst hold RMS [deg]": f"{hold.max():.4f}",
            "runs with momentum violation %": f"{hv.mean() * 100:.1f}",
        })
    df = pd.DataFrame(tab)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(f"results/tables/robustness_stage{a.stage}.csv", index=False)
    Path(f"results/tables/robustness_stage{a.stage}.md").write_text(
        df.to_markdown(index=False) + f"\n\n{a.n} sampled plants (seed 777), identical for every design.\n", encoding="utf-8")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
