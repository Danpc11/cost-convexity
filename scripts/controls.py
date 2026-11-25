#!/usr/bin/env python
"""Controls on the revised states (Supplemental Material, Sec. S1).

  threshold   rerun the search with truncation disabled, then refine the
              same pretruncation solution at cuts 1e-4, 1e-5, 1e-6
  insertion   open every closed edge of a reported star by a finite share and refine
  seeds       repeat a state with three independent random start sets
  budget      raise the number of Dirichlet starts from 24 to 64 and 144

  python scripts/controls.py --out results/controls
"""
import argparse
import os
import sys
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from costconvexity import core, starts  # noqa: E402


def full_starts(n, E, seed, n_random):
    rng = np.random.default_rng(seed)
    s, lab = starts.legacy_starts(n, E, rng, n_random)
    ls, ll = starts.looped_starts(n, E, seed=seed + 12345)
    ts, tl = starts.tree_starts(n, E, seed=seed + 54321)
    return s + ls + ts, lab + ll + tl


def K(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def threshold(cases=((0.55, 32), (0.65, 32), (0.75, 64), (0.90, 64))):
    rows = []
    for a, n in cases:
        E = K(n)
        prob = core.Problem(n, E, a)
        s, lab = full_starts(n, E, 7, 24)
        raw = core.multistart(prob, s, lab, prune=False)
        for cut in (1e-4, 1e-5, 1e-6):
            ref = core.refine_support(prob, raw["p"], cut=cut)
            st = core.summarize_state(prob, ref["p"], ref["w"], cut)
            kept = ref["p"][ref["p"] > 0]
            rows.append(dict(alpha=a, n=n, cut=cut, F=ref["value"], active=st["active"],
                             cycle_rank=st["cycle_rank"], min_share=float(kept.min()),
                             eff_support=st["eff_support"]))
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def insertion(alpha=0.64, n=32, eps=(0.01, 0.02, 0.05)):
    E = K(n)
    prob = core.Problem(n, E, alpha)
    s, lab = full_starts(n, E, 7, 24)
    base = core.multistart(prob, s, lab)
    rows = []
    closed = np.nonzero(base["p"] <= 1e-5)[0]
    for e in eps:
        best = np.inf
        improving = 0
        for k in closed:
            p = base["p"].copy() * (1 - e)
            p[k] = e
            r = core.refine_support(prob, p, cut=1e-5)
            v = r["value"] if r is not None else np.inf
            best = min(best, v)
            improving += v < base["best"] * (1 - 1e-9)
        rows.append(dict(alpha=alpha, n=n, eps=e, closed=len(closed), improving=int(improving),
                         base=base["best"], best_after=best,
                         rel_gap=(best - base["best"]) / base["best"], base_beta=base["cycle_rank"]))
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def seeds(alpha=0.66, n=32):
    rows = []
    E = K(n)
    for sd in (7, 101, 2029):
        s, lab = full_starts(n, E, sd, 24)
        r = core.multistart(core.Problem(n, E, alpha), s, lab)
        rows.append(dict(alpha=alpha, n=n, seed=sd, F=r["best"], active=r["active"],
                         cycle_rank=r["cycle_rank"], winner=r["winner"]))
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def budget(alpha=0.64, n=32):
    rows = []
    E = K(n)
    for nr in (24, 64, 144):
        s, lab = full_starts(n, E, 7, nr)
        r = core.multistart(core.Problem(n, E, alpha), s, lab)
        rows.append(dict(alpha=alpha, n=n, n_random=nr, F=r["best"], active=r["active"],
                         cycle_rank=r["cycle_rank"]))
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/controls")
    ap.add_argument("--only", default="threshold,insertion,seeds,budget")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in a.only.split(","):
        globals()[name]().to_csv(out / f"{name}.tsv", sep="\t", index=False)
