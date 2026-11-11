"""Summarize one stage's histories into tables + convergence figures.

Usage: python -m scripts.summarize_stage --stage A
Reads results/histories/stage<S>/*.json, writes
  results/stage<S>_summary.json
  results/tables/stage<S>_main.{csv,md}
  figures/convergence_stage<S>.{png,pdf}, figures/first_feasible_stage<S>.{png,pdf}
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from optim.common import best_feasible, best_so_far_curve, first_feasible_iter
from scripts.style import COL_W, COLORS, LABELS, METHOD_ORDER, save, setup

FIXED = {"pidtune", "analytic"}
REACH_FRAC = 1.10   # "evaluations to reach within 10% of the best design found by any method"


def load(stage):
    runs = defaultdict(list)
    for p in sorted(Path(f"results/histories/stage{stage}").glob("*.json")):
        r = json.loads(p.read_text())
        runs[r["method"]].append(r)
    for m in runs:
        runs[m].sort(key=lambda r: r["seed"])
    return runs


def _stats(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return dict(n=0, mean=None, std=None, median=None, q25=None, q75=None, min=None, max=None)
    return dict(n=int(v.size), mean=float(v.mean()), std=float(v.std(ddof=1)) if v.size > 1 else 0.0,
                median=float(np.median(v)), q25=float(np.percentile(v, 25)),
                q75=float(np.percentile(v, 75)), min=float(v.min()), max=float(v.max()))


def summarize(stage):
    runs = load(stage)
    # reference value: best feasible objective found by any run of any method
    all_best = [best_feasible(r["history"]) for rs in runs.values() for r in rs]
    ref = min(b["objective"] for b in all_best if b is not None)
    out = {"stage": stage, "reference_best_t_settle": ref, "reach_threshold": ref * REACH_FRAC,
           "methods": {}}
    for m in [x for x in METHOD_ORDER if x in runs]:
        rs = runs[m]
        bests = [best_feasible(r["history"]) for r in rs]
        ff = [first_feasible_iter(r["history"]) for r in rs]
        reach, beat = [], []
        pid = runs.get("pidtune")
        pid_t = best_feasible(pid[0]["history"])["objective"] if pid and best_feasible(pid[0]["history"]) else None
        for r in rs:
            c0 = best_so_far_curve(r["history"], None)
            ib = np.nonzero(c0 < pid_t)[0] if pid_t is not None else np.array([])
            beat.append(int(ib[0]) + 1 if ib.size else None)
            c = best_so_far_curve(r["history"], None)
            idx = np.nonzero(c <= ref * REACH_FRAC)[0]
            reach.append(int(idx[0]) + 1 if idx.size else None)
        out["methods"][m] = {
            "n_runs": len(rs),
            "budget": rs[0]["budget"] if m not in FIXED else 0,
            "seeds": [r["seed"] for r in rs],
            "found_feasible_frac": float(np.mean([b is not None for b in bests])),
            "best_t_settle": _stats([b["objective"] if b else None for b in bests]),
            "best_t_settle_per_seed": [b["objective"] if b else None for b in bests],
            "best_gains_per_seed": [b["gains"] if b else None for b in bests],
            "best_metrics_per_seed": [b["metrics"] if b else None for b in bests],
            "evals_to_first_feasible": _stats(ff),
            "evals_to_first_feasible_per_seed": ff,
            "evals_to_reach": _stats(reach),
            "reach_frac_runs": float(np.mean([x is not None for x in reach])),
            "pidtune_t_settle": pid_t,
            "evals_to_beat_pidtune": _stats(beat),
            "evals_to_beat_pidtune_per_seed": beat,
            "beat_pidtune_frac_runs": float(np.mean([x is not None for x in beat])),
            "feasible_eval_frac": float(np.mean([h["feasible"] for r in rs for h in r["history"]])),
            "wall_s_mean": float(np.mean([r["wall_s"] for r in rs])),
            "fixed_design_metrics": rs[0]["history"][0]["metrics"] if m in FIXED else None,
        }
    Path("results").mkdir(exist_ok=True)
    Path(f"results/stage{stage}_summary.json").write_text(json.dumps(out, indent=2))
    return out, runs


def fmt(s, nd=1):
    if s["n"] == 0:
        return "—"
    return f"{s['mean']:.{nd}f} ± {s['std']:.{nd}f}"


def table(out, stage):
    rows = []
    for m, d in out["methods"].items():
        fixed = m in FIXED
        bt = d["best_t_settle"]
        rows.append({
            "method": LABELS[m],
            "budget": d["budget"],
            "runs": d["n_runs"],
            "runs finding a feasible design": f"{d['found_feasible_frac'] * 100:.0f}%",
            "best feasible t_settle [s] (mean ± std)": ("infeasible" if bt["n"] == 0 else
                                                       f"{bt['mean']:.2f}" if fixed else fmt(bt, 2)),
            "median [IQR]": "—" if fixed or bt["n"] == 0 else
            f"{bt['median']:.2f} [{bt['q25']:.2f}, {bt['q75']:.2f}]",
            "evals to first feasible (median)": "—" if fixed else
            ("never" if d["evals_to_first_feasible"]["n"] == 0 else f"{d['evals_to_first_feasible']['median']:.0f}"),
            "evals to beat pidtune (median; runs that do)": "—" if fixed else
            ("never" if d["evals_to_beat_pidtune"]["n"] == 0 else
             f"{d['evals_to_beat_pidtune']['median']:.0f} ({d['beat_pidtune_frac_runs'] * 100:.0f}%)"),
            f"evals to within 10% of best (median; runs reaching)": "—" if fixed else
            ("never" if d["evals_to_reach"]["n"] == 0 else
             f"{d['evals_to_reach']['median']:.0f} ({d['reach_frac_runs'] * 100:.0f}%)"),
            "feasible fraction of all evals": "—" if fixed else f"{d['feasible_eval_frac'] * 100:.0f}%",
        })
    df = pd.DataFrame(rows)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(f"results/tables/stage{stage}_main.csv", index=False)
    md = df.to_markdown(index=False)
    note = (f"\n\nReference best (any method, any seed): {out['reference_best_t_settle']:.2f} s; "
            f"'within 10%' threshold = {out['reach_threshold']:.2f} s. Fixed designs (pidtune, analytic) "
            f"use zero optimization budget and are a single deterministic evaluation.\n")
    Path(f"results/tables/stage{stage}_main.md").write_text(md + note)
    return md + note


def figures(out, runs, stage):
    plt = setup()
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for m in [x for x in METHOD_ORDER if x in runs and x not in FIXED]:
        C = np.array([best_so_far_curve(r["history"], None) for r in runs[m]])
        n = C.shape[1]
        x = np.arange(1, n + 1)
        with np.errstate(all="ignore"):
            # median over seeds; seeds without a feasible point yet count as +inf (shown as gap)
            Cf = np.where(np.isnan(C), np.inf, C)
            med = np.median(Cf, axis=0)
            lo = np.percentile(Cf, 25, axis=0)
            hi = np.percentile(Cf, 75, axis=0)
        med[~np.isfinite(med)] = np.nan
        hi[~np.isfinite(hi)] = np.nan
        lo[~np.isfinite(lo)] = np.nan
        ax.plot(x, med, color=COLORS[m], label=LABELS[m])
        ax.fill_between(x, lo, hi, color=COLORS[m], alpha=0.15, lw=0)
    for m in FIXED & set(runs):
        b = best_feasible(runs[m][0]["history"])
        if b:
            ax.axhline(b["objective"], color=COLORS[m], ls="--", lw=1, label=LABELS[m] + " (fixed)")
    ax.set_yscale("log")
    ax.set_xlabel("evaluations")
    ax.set_ylabel("best feasible $t_{settle}$ (s)")
    ax.set_title(f"Stage {stage}: median and IQR over seeds")
    ax.legend(fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)
    save(fig, f"figures/convergence_stage{stage}")

    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    ms = [x for x in METHOD_ORDER if x in runs and x not in FIXED]
    data = []
    for m in ms:
        ff = out["methods"][m]["evals_to_first_feasible_per_seed"]
        budget = out["methods"][m]["budget"]
        data.append([f if f is not None else budget + 1 for f in ff])
    ax.boxplot(data, vert=False, tick_labels=[LABELS[m] for m in ms], widths=0.5)
    for i, m in enumerate(ms):
        ax.scatter(data[i], np.full(len(data[i]), i + 1) + np.random.default_rng(0).uniform(-0.15, 0.15, len(data[i])),
                   s=6, color=COLORS[m], zorder=3)
    ax.set_xlabel("evaluations to first feasible design (budget+1 = never)")
    save(fig, f"figures/first_feasible_stage{stage}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="A")
    a = ap.parse_args()
    out, runs = summarize(a.stage)
    print(table(out, a.stage))
    figures(out, runs, a.stage)


if __name__ == "__main__":
    main()
