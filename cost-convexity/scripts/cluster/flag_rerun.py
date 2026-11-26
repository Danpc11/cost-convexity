#!/usr/bin/env python
"""Mark for rerun the sweeps that the small-conductance bug of v2.0.0-rc3 could
have affected, so that run_pool.py repeats them.

Before the fix in core.Problem.spectral, a Laplacian whose eigenvalues all lay
below 1e-13 was rejected as singular, and objective values above 1e25 were taken
as infeasible. Both happen only when the typical conductance
w ~ (1/m)^(1/alpha) is tiny, i.e. on large candidate graphs at small alpha
(lattices and random graphs below alpha ~ 0.3). A sweep is flagged when, at the
smallest exponent it visited, mean_degree * (1/m)^(1/alpha) < 1e-9 (a wide
margin) or when its objective exceeded 1e20 anywhere. Flagged result files are
moved to results/_suspect_rc3/ (not deleted), so their summaries disappear and the
pool reruns them. The complete-graph sweeps (alpha >= 0.45) are never affected.

  python scripts/cluster/flag_rerun.py            # report only
  python scripts/cluster/flag_rerun.py --move     # move flagged files
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--move", action="store_true")
a = ap.parse_args()
res = Path("results")
dest = res / "_suspect_rc3"
flagged = []
for f in sorted(res.glob("E*/*_summary.json")):
    d = json.loads(f.read_text())
    tsv = f.with_name(f.name.replace("_summary.json", ".tsv"))
    if not tsv.exists():
        continue
    t = pd.read_csv(tsv, sep="\t")
    a_min = float(t.alpha.min())
    m = int(d["m"])
    kbar = 2.0 * m / int(d["n"])
    w_typ = (1.0 / m) ** (1.0 / a_min)
    risky = kbar * w_typ < 1e-9 or float(t.F.max()) > 1e20
    if risky:
        flagged.append((f.parent.name, f.name.replace("_summary.json", ""), a_min, m,
                        kbar * w_typ, float(t.F.max())))
print(f"{len(flagged)} sweeps flagged")
for e, tag, a_min, m, lam, fmax in flagged:
    print(f"  {e}/{tag}: alpha_min={a_min:.2f} m={m} lambda_typ={lam:.1e} F_max={fmax:.1e}")
if a.move:
    for e, tag, *_ in flagged:
        (dest / e).mkdir(parents=True, exist_ok=True)
        for suffix in (".tsv", "_shares.npz", "_summary.json"):
            g = res / e / f"{tag}{suffix}"
            if g.exists():
                shutil.move(str(g), dest / e / g.name)
    print(f"moved to {dest}; relaunch run_pool.py to recompute them")
