"""Final comparison figures across PID (stage B) and expert-law (stage W) designs.

  figures/robustness_compliance   spec compliance of each design over 200 plants
  figures/robustness_tsettle      t_settle distribution over plants, pooled per method
  figures/large_slew              90 deg slew: settled / overshoot / momentum-limit hit rates
  figures/slew_responses          nominal 30 deg and 90 deg responses of representative designs
Usage: python -m scripts.figures_final
"""
import json
from pathlib import Path

import numpy as np

from scripts.style import COL_W, DBL_W, METHOD_ORDER, color, label, save, setup

SHOW_B = ["bo_robust", "bo", "tpe_iecon", "cmaes", "random_search", "pidtune"]
SHOW_W = ["bo", "wie_rule"]


def _keys(present):
    ks = [("W", m) for m in SHOW_W] + [("B", m) for m in SHOW_B]
    return [k for k in ks if k in present]


def robustness(plt):
    D = {}
    for st in ("B", "W"):
        f = Path(f"results/robustness_stage{st}.json")
        if f.exists():
            for d in json.loads(f.read_text(encoding="utf-8"))["designs"]:
                D.setdefault((st, d["method"]), []).append(d)
    ks = _keys(D)
    fig, ax = plt.subplots(figsize=(COL_W, 0.32 * len(ks) + 0.6))
    rng = np.random.default_rng(0)
    for i, k in enumerate(ks):
        c = [np.mean([r["compliant"] for r in d["plants"]]) * 100 for d in D[k]]
        y = len(ks) - i
        ax.scatter(c, y + rng.uniform(-0.15, 0.15, len(c)), s=9, color=color(*k), zorder=3)
        ax.plot([np.mean(c)] * 2, [y - 0.3, y + 0.3], color="k", lw=1.2)
    ax.set_yticks(range(len(ks), 0, -1), [label(*k) for k in ks])
    ax.set_xlabel("spec compliance over 200 sampled plants (%)\n(dot = one tuned design / seed; bar = mean)")
    ax.set_xlim(-2, 102)
    save(fig, "figures/robustness_compliance")

    fig, ax = plt.subplots(figsize=(COL_W, 0.32 * len(ks) + 0.6))
    data = [np.array([r["t_settle"] for d in D[k] for r in d["plants"]]) for k in ks]
    bp = ax.boxplot(data[::-1], vert=False, widths=0.55, whis=(5, 95), showfliers=False, patch_artist=True)
    for patch, k in zip(bp["boxes"], ks[::-1]):
        patch.set_facecolor(color(*k))
        patch.set_alpha(0.35)
    ax.set_yticks(range(1, len(ks) + 1), [label(*k) for k in ks[::-1]])
    ax.set_xscale("log")
    ax.set_xlabel("30° slew $t_{settle}$ over plants (s); whiskers 5–95th pct")
    save(fig, "figures/robustness_tsettle")


def large_slew(plt):
    f = Path("results/large_slew.json")
    if not f.exists():
        return
    D = {}
    for d in json.loads(f.read_text(encoding="utf-8"))["designs"]:
        D.setdefault((d["stage"], d["method"]), []).append(d)
    ks = _keys(D)
    fig, axs = plt.subplots(1, 3, figsize=(DBL_W, 0.3 * len(ks) + 0.8), sharey=True)
    titles = ["settled within 200 s (%)", "overshoot ≤ 5% (%)", "wheel momentum hit limit (%)"]
    fns = [lambda r: r["settled"], lambda r: r["overshoot"] <= 5.0, lambda r: r["h_violation"]]
    for ax, t, fn in zip(axs, titles, fns):
        v = [np.mean([fn(r) for d in D[k] for r in d["plants"]]) * 100 for k in ks]
        ax.barh(range(len(ks), 0, -1), v, color=[color(*k) for k in ks])
        ax.set_title(t)
        ax.set_xlim(0, 100)
    axs[0].set_yticks(range(len(ks), 0, -1), [label(*k) for k in ks])
    fig.suptitle("90° slew (never seen during tuning), 200 sampled plants", y=1.02)
    save(fig, "figures/large_slew")


def responses(plt):
    from cubesat.params import DEG, LARGE_SLEW, SLEW, EnvParams, SimParams, SpacecraftParams
    from cubesat.three_axis import run_three_axis
    picks = []
    for st, m in [("W", "bo"), ("W", "wie_rule"), ("B", "bo"), ("B", "bo_robust"), ("B", "pidtune")]:
        f = Path(f"results/stage{st}_summary.json")
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))["methods"].get(m)
        if not d:
            continue
        ts = [(t, g) for t, g in zip(d["best_t_settle_per_seed"], d["best_gains_per_seed"]) if t is not None]
        ts.sort(key=lambda x: x[0])
        picks.append((st, m, ts[len(ts) // 2][1]))   # median seed: representative, not best-case
    fig, ax = plt.subplots(3, 2, figsize=(DBL_W, 5.0), sharex="col")
    for col, sc_ in enumerate((SLEW, LARGE_SLEW)):
        for st, m, g in picks:
            r = run_three_axis(np.asarray(g), sc_, SpacecraftParams(), EnvParams(), SimParams(), 3000,
                               law=1 if st == "W" else 0)
            ax[0, col].plot(r.t, r.err / DEG, color=color(st, m), label=label(st, m))
            ax[1, col].plot(r.t, np.linalg.norm(r.rate, axis=1) / DEG, color=color(st, m), lw=0.9)
            ax[2, col].plot(r.t, np.abs(r.h).max(axis=1) * 1e3, color=color(st, m), lw=0.9)
        ax[0, col].set_yscale("log")
        ax[0, col].set_ylim(1e-3, 200)
        ax[0, col].axhline(0.1, color="k", ls=":", lw=0.8)
        ax[2, col].axhline(SpacecraftParams().h_max * 1e3, color="k", ls=":", lw=0.8)
        ax[0, col].set_title(f"{sc_.angle_deg:.0f}° slew, nominal plant")
        ax[2, col].set_xlabel("time (s)")
        ax[0, col].set_xlim(0, 60 if col == 0 else 120)
    ax[0, 0].set_ylabel("pointing error (deg)")
    ax[1, 0].set_ylabel("|body rate| (deg/s)")
    ax[2, 0].set_ylabel("max wheel |h| (mN m s)")
    ax[0, 0].legend(fontsize=6, loc="upper right")
    fig.align_ylabels(ax[:, 0])
    save(fig, "figures/slew_responses")


def main():
    plt = setup()
    robustness(plt)
    large_slew(plt)
    responses(plt)


if __name__ == "__main__":
    main()
