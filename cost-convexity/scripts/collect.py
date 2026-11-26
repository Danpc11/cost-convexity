#!/usr/bin/env python
"""Gather sweep summaries and draw the revision figures.

  results/summary_alpha_star.tsv      one row per sweep (graph, load, n, alpha*, ...)
  results/figures/alpha_star_complete.png   alpha*(n) on K_n: v1, revised, certified bound
  results/figures/alpha_star_topology.png   alpha*(n) by candidate topology
  results/figures/alpha_star_loads.png      alpha* against the load mixture
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

V1 = {8: .555, 12: .585, 16: .61, 20: .63, 24: .635, 28: .645, 32: .65,
      40: .66, 48: .665, 56: .67, 64: .675}

rows = []
for f in sorted(RES.glob("E*/**/*_summary.json")):
    d = json.loads(f.read_text())
    d["experiment"] = f.parent.name
    d["family"] = d["graph"].split(":")[0]
    rows.append(d)
S = pd.DataFrame(rows)
if len(S):
    S.to_csv(RES / "summary_alpha_star.tsv", sep="\t", index=False)
    print(S[["experiment", "graph", "load", "n", "alpha_star", "bracketed",
             "active_above", "beta_above", "mean_degree_above"]].to_string(index=False))

# ---- Fig. A: complete graph
fig, ax = plt.subplots(figsize=(5.0, 3.6))
ns = np.unique(np.r_[np.geomspace(8, 4096, 80).astype(int), list(V1)])
ax.plot(ns, np.log(ns / 2) / np.log(ns - 1), "-", color="0.5", label=r"$\alpha_c(n)$ (star vs complete)")
ax.plot(list(V1), list(V1.values()), "o", mfc="none", color="0.4", label="version 1 (blind search)")
B = RES / "E4" / "ansatz_bound.tsv"
if B.exists():
    b = pd.read_csv(B, sep="\t")
    b["best"] = b[["alpha_unif", "alpha_opt"]].min(axis=1)
    g = b.groupby("n")["best"].min()
    ax.plot(g.index, g.values, "-", color="C3", label="certified upper bound (regular competitors)")
    ax.axhline(b.alpha_km_limit.min(), ls=":", color="C3", lw=1)
if len(S):
    e1 = S[(S.experiment == "E1") & (S.load == "iso")].sort_values("n")
    ax.plot(e1.n, e1.alpha_star, "s", color="C0", label="revised search")
ax.set_xscale("log")
ax.set_xlabel("n")
ax.set_ylabel(r"loop-collapse exponent $\alpha^*$")
ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "alpha_star_complete.png", dpi=200)

# ---- Fig. B: topology
if len(S):
    e2 = S[(S.load == "iso") & S.experiment.isin(["E1", "E2"])].copy()
    e2["fam"] = e2.graph.str.replace(r":\d+$", "", regex=True).where(
        ~e2.family.isin(["rrg", "er", "ba"]),
        e2.graph.str.split(":").str[:2].str.join(":"))
    e2.loc[e2.family.isin(["complete", "square", "triangular"]), "fam"] = e2.family
    e2.loc[e2.family == "rgg", "fam"] = "rgg"
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for fam, g in e2.groupby("fam"):
        g = g.sort_values("n")
        ax.plot(g.n, g.alpha_star, "o-", ms=3, label=fam)
    ax.set_xscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel(r"$\alpha^*$")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "alpha_star_topology.png", dpi=200)

# ---- Fig. C: loads
if len(S):
    e3 = S[S.experiment == "E3"].copy()
    if len(e3):
        e3["lam"] = e3.load.str.extract(r"mix:([0-9.]+)").astype(float)
        e1i = S[(S.load == "iso") & S.family.isin(["complete", "square"])].copy()
        e1i["lam"] = 1.0
        mix = pd.concat([e3[e3.lam.notna()], e1i])
        # all-tree sweeps report alpha* >= top of grid
        mix["a"] = mix.alpha_star.fillna(mix.get("alpha_star_lower_bound"))
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        for (fam, n), g in mix.groupby(["family", "n"]):
            g = g.sort_values("lam")
            ax.plot(g.lam, g.a, "o-" if fam == "complete" else "s--", ms=3, label=f"{fam} n={n}")
        ax.set_xlabel(r"isotropic fraction $\lambda$ of the load")
        ax.set_ylabel(r"$\alpha^*$ (top of grid if no loops)")
        ax.legend(fontsize=6, frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(FIG / "alpha_star_loads.png", dpi=200)
print("figures in", FIG)
