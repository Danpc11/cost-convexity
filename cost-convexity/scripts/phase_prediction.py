#!/usr/bin/env python
"""Uniform-weight phase prediction on K_n: for each (n, alpha), the state among
the star, low-resistance k-regular graphs and the complete graph that minimizes
A(H) m_H^(1/alpha). Used for the phase-diagram panel (boundaries between the star
and the k-regular phases), to be compared with the E6 scan.

  python scripts/phase_prediction.py --out results/E4/phase_prediction.tsv
"""
import argparse, os, sys
from pathlib import Path
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from costconvexity import starts

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", default="8,12,16,24,32,48,64,96,128,192,256,384,512")
ap.add_argument("--ks", default="3,4,5,6,7,8,9,10,11,12,14,16,20,24,32,48,64")
ap.add_argument("--out", default="results/E4/phase_prediction.tsv")
a = ap.parse_args()
alphas = np.round(np.arange(0.50, 0.9951, 0.0025), 4)
rows, graphs = [], []
for n in [int(x) for x in a.sizes.split(",")]:
    ks = tuple(k for k in (int(x) for x in a.ks.split(",")) if k < n - 1)
    G = starts.best_regular_graphs(n, ks, samples=4 if n <= 128 else 2, seed=n, descent=n <= 64)
    cand = [("star", 1, n - 1, 2 * (n - 1) / n)]
    cand += [(f"reg{k}", k, n * k // 2, A) for k, (A, _) in sorted(G.items())]
    cand += [("complete", n - 1, n * (n - 1) // 2, 2.0 / n)]
    for lab, k, m, A in cand:
        graphs.append(dict(n=n, label=lab, k=k, m=m, A=A))
    for al in alphas:
        v = [np.log(A) + np.log(m) / al for _, _, m, A in cand]
        i = int(np.argmin(v))
        rows.append(dict(n=n, alpha=al, label=cand[i][0], k=cand[i][1], m=cand[i][2]))
    print(f"n={n} done", flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(a.out, sep="\t", index=False)
pd.DataFrame(graphs).to_csv(Path(a.out).with_name("phase_prediction_graphs.tsv"), sep="\t", index=False)
