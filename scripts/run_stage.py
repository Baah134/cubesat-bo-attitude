"""Run every optimizer x seed for one stage and save full histories.

Usage: python -m scripts.run_stage --config configs/stageA.yaml [--workers 10]
Outputs: results/histories/<stage>/<method>_s<seed>.json (one per run)
Each job is independent and deterministic given (method, seed); finished
jobs are skipped on re-run unless --force.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml


def _job(stage, method, seed, budget, out_path, specs):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import warnings
    warnings.filterwarnings("ignore")
    import importlib

    import torch
    torch.set_num_threads(1)
    from cubesat.problem import Problem, Specs
    p = Problem(stage=stage, specs=Specs(**specs))
    mod = importlib.import_module(f"optim.{method}")
    t0 = time.perf_counter()
    hist = mod.optimize(p, budget, seed)
    rec = {"stage": stage, "method": method, "seed": seed, "budget": budget,
           "specs": specs, "wall_s": time.perf_counter() - t0, "history": hist}
    Path(out_path).write_text(json.dumps(rec))
    return method, seed, rec["wall_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--methods", nargs="*")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    stage = cfg["stage"]
    out_dir = Path("results/histories") / f"stage{stage}"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for method in a.methods or cfg["methods"]:
        seeds = [0] if method in cfg.get("fixed_designs", []) else list(range(cfg["n_seeds"]))
        for s in seeds:
            out = out_dir / f"{method}_s{s}.json"
            if out.exists() and not a.force:
                continue
            jobs.append((stage, method, s, cfg["budget"], str(out), cfg.get("specs", {})))
    print(f"{len(jobs)} jobs, {a.workers} workers", flush=True)
    # BO jobs are slowest: submit them first
    jobs.sort(key=lambda j: j[1] != "bo")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_job, *j) for j in jobs]
        for f in as_completed(futs):
            m, s, w = f.result()
            print(f"done {m} seed {s} in {w:.1f}s", flush=True)


if __name__ == "__main__":
    main()
