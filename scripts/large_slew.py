"""Generalization test: every tuned design (tuned on the 30 deg slew) flown on a
90 deg slew it never saw, on the nominal plant and on the same 200 sampled plants.

The 90 deg slew is where reaction-wheel momentum becomes nearly binding: per axis
(error component theta/sqrt(3), alpha = tau_max/J) the time-optimal bang-bang profile
peaks at w = sqrt(alpha*theta/sqrt(3)) = 0.21 rad/s, i.e. J*w = 8.7 mN m s per wheel,
87% of h_max = 10 mN m s. A controller without rate limiting that drives the body
faster than that profile will hit the wheel momentum limit here.

Usage: python -m scripts.large_slew [--n 200] [--stages B W]
Writes results/large_slew.json, results/tables/large_slew.{csv,md}.
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.style import METHOD_ORDER, label


def _eval(args):
    stage, gains, n = args
    import warnings
    warnings.filterwarnings("ignore")
    from cubesat.params import EnvParams, LARGE_SLEW, SimParams, SpacecraftParams
    from cubesat.simulate import scenario_metrics
    from cubesat.three_axis import run_three_axis
    from cubesat.uncertainty import sample_plants
    law = 1 if stage == "W" else 0
    g = np.asarray(gains)
    rows = []
    plants = [(-1, SpacecraftParams(), EnvParams(), 3000)] + \
             [(s.index, s.sc, s.env, s.slew_seed) for s in sample_plants(n)]
    for idx, sc, env, seed in plants:
        r = run_three_axis(g, LARGE_SLEW, sc, env, SimParams(), seed, law=law)
        m = scenario_metrics(r, LARGE_SLEW, sc)
        rows.append({"plant": idx, "t_settle": float(m["t_settle"]),
                     "settled": bool(m["t_settle"] < LARGE_SLEW.horizon and not m["diverged"]),
                     "overshoot": float(min(m["overshoot"], 200.0)), "h_peak_frac": float(m["h_peak_frac"]),
                     "h_violation": bool(m["h_violation"]), "sat_frac": float(m["sat_frac"]),
                     "diverged": bool(m["diverged"])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--stages", nargs="*", default=["B", "W"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    designs = []
    for st in a.stages:
        f = Path(f"results/stage{st}_summary.json")
        if not f.exists():
            print(f"skip stage {st}: no summary")
            continue
        summ = json.loads(f.read_text(encoding="utf-8"))
        for m in [x for x in METHOD_ORDER if x in summ["methods"]]:
            d = summ["methods"][m]
            for seed, g in zip(d["seeds"], d["best_gains_per_seed"]):
                if g is not None:
                    designs.append((st, m, seed, g))
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(_eval, [(st, g, a.n) for st, _, _, g in designs]))
    out = {"scenario": "large_slew_90deg", "n_plants": a.n, "plant_seed": 777, "designs": []}
    for (st, m, seed, g), rows in zip(designs, res):
        out["designs"].append({"stage": st, "method": m, "seed": seed, "gains": g,
                               "nominal": rows[0], "plants": rows[1:]})
    Path("results/large_slew.json").write_text(json.dumps(out), encoding="utf-8")

    tab = []
    for key in dict.fromkeys((d["stage"], d["method"]) for d in out["designs"]):
        ds = [d for d in out["designs"] if (d["stage"], d["method"]) == key]
        nom = [d["nominal"] for d in ds]
        P = [r for d in ds for r in d["plants"]]
        ts = np.array([r["t_settle"] for r in P if r["settled"]])
        tab.append({
            "method": label(*key), "designs": len(ds),
            "nominal: settled": f"{np.mean([r['settled'] for r in nom]) * 100:.0f}%",
            "nominal: t_settle median [s]": f"{np.median([r['t_settle'] for r in nom]):.1f}",
            "nominal: overshoot median [%]": f"{np.median([r['overshoot'] for r in nom]):.1f}",
            "plants: settled": f"{np.mean([r['settled'] for r in P]) * 100:.1f}%",
            "plants: overshoot <= 5%": f"{np.mean([r['overshoot'] <= 5 for r in P]) * 100:.1f}%",
            "plants: wheel momentum hit limit": f"{np.mean([r['h_violation'] for r in P]) * 100:.1f}%",
            "plants: t_settle median, settled runs [s]": f"{np.median(ts):.1f}" if ts.size else "—",
        })
    df = pd.DataFrame(tab)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/tables/large_slew.csv", index=False)
    Path("results/tables/large_slew.md").write_text(
        df.to_markdown(index=False) + f"\n\n90 deg slew about [1,1,1]/sqrt(3), 200 s horizon. Designs were tuned "
        f"on the 30 deg slew only. Nominal plant plus {a.n} sampled plants (seed 777).\n", encoding="utf-8")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
