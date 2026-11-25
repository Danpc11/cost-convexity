#!/usr/bin/env python
"""Certified upper bounds on alpha*(n) for the complete candidate graph.

For each n and degree k a low-resistance k-regular graph H is built (best of
several random k-regular graphs, improved by degree-preserving swap descent up
to n = 256). Two crossings with the optimal tree (the star, ansatz.py) are
reported:

  alpha_unif   uniform conductance on H beats the star for alpha > alpha_unif
  alpha_opt    conductances optimized on the fixed support H beat the star for
               alpha > alpha_opt (bisection; computed up to n = n_opt_max)

Both are certificates: an explicit feasible state that beats the best tree
shows the global optimum is not a tree, so alpha*(n) <= min_k alpha(n).
"""
import argparse, os, sys, time
from pathlib import Path
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from costconvexity import ansatz, starts

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", default="8,12,16,20,24,28,32,40,48,56,64,80,96,128,192,256,384,512,768,1024,1536,2048")
ap.add_argument("--ks", default="3,4,5,6,7,8")
ap.add_argument("--n-opt-max", type=int, default=128)
ap.add_argument("--out", default="results/E4/ansatz_bound.tsv")
a = ap.parse_args()
ks = tuple(int(k) for k in a.ks.split(","))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
rows = []
for n in [int(x) for x in a.sizes.split(",")]:
    t = time.time()
    samples = 200 if n <= 32 else 40 if n <= 128 else 8 if n <= 512 else 2
    descent = n <= 256
    graphs = starts.best_regular_graphs(n, ks, samples=samples, seed=n, descent=descent)
    for k, (A, H) in sorted(graphs.items()):
        m = n * k // 2
        au = ansatz.crossing_alpha(n, A, m)
        ao = np.nan
        if n <= a.n_opt_max and np.isfinite(au):
            lo, hi = 0.40, au
            for _ in range(14):
                mid = 0.5 * (lo + hi)
                _, R_opt, R_star = ansatz.certify(n, H, mid)
                if R_opt < R_star:
                    hi = mid
                else:
                    lo = mid
            ao = hi
        rows.append(dict(n=n, k=k, m=m, A=A, alpha_unif=au, alpha_opt=ao,
                         alpha_c=np.log(n / 2) / np.log(n - 1),
                         A_km=2 * (k - 1) / (k * (k - 2)), alpha_km_limit=ansatz.km_limit(k),
                         samples=samples, swap_descent=descent))
        print(f"n={n:5d} k={k} A={A:.5f} alpha_unif={au:.4f} alpha_opt={ao:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(a.out, sep="\t", index=False)
    print(f"  n={n} done in {time.time()-t:.0f}s", flush=True)
