#!/usr/bin/env python
"""One alpha*(n) sweep: one candidate graph, one load, a descending grid of exponents.

Example:
  python scripts/run_sweep.py --graph complete:64 --load iso --amax 0.72 --amin 0.50 \
      --step 0.005 --workers 32 --out results/E1
"""
import argparse
import os
import sys
from pathlib import Path

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from costconvexity import sweep  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--graph", required=True)
ap.add_argument("--load", default="iso")
ap.add_argument("--amax", type=float, default=0.75)
ap.add_argument("--amin", type=float, default=0.30)
ap.add_argument("--step", type=float, default=0.005)
ap.add_argument("--n-random", type=int, default=24)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--confirm", type=int, default=4)
ap.add_argument("--looped-samples", type=int, default=40)
ap.add_argument("--no-looped", action="store_true")
ap.add_argument("--no-upward", action="store_true")
ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
ap.add_argument("--grid", default=None, help="comma-separated exponents (overrides amax/amin/step)")
ap.add_argument("--out", default="results")
ap.add_argument("--tag", default=None)
a = ap.parse_args()

sweep.sweep(a.graph, a.load, a.amax, a.amin, a.step, a.n_random, a.seed, a.confirm,
            looped=not a.no_looped, looped_samples=a.looped_samples, workers=a.workers,
            outdir=a.out, tag=a.tag, upward=not a.no_upward,
            grid=None if a.grid is None else [float(x) for x in a.grid.split(",")])
