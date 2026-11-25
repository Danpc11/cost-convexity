#!/usr/bin/env python
"""Write the Supplemental Tables S1-S5 and the control tables as LaTeX.

  python scripts/sm_tables.py --res results --out supplement/sm_tables.tex
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--res", default="results")
ap.add_argument("--out", default="supplement/sm_tables.tex")
a = ap.parse_args()
RES = Path(a.res)
V1 = {8: .555, 12: .585, 16: .61, 20: .63, 24: .635, 28: .645, 32: .65,
      40: .66, 48: .665, 56: .67, 64: .675}
out = []


def table(label, caption, cols, rows, align=None, star=False):
    env = "table*" if star else "table"
    align = align or "r" * len(cols)
    num = int(label.split("S")[-1])
    out.append(f"\\setcounter{{table}}{{{num - 1}}}\n\\begin{{{env}}}[p]\n\\caption{{\\label{{{label}}}{caption}}}\n"
               f"\\begin{{ruledtabular}}\n\\begin{{tabular}}{{{align}}}\n"
               + " & ".join(cols) + "\\\\\n\\hline\n"
               + "\n".join(" & ".join(str(x) for x in r) + "\\\\" for r in rows)
               + f"\n\\end{{tabular}}\n\\end{{ruledtabular}}\n\\end{{{env}}}\n")


def summaries(exp):
    rows = [json.loads(f.read_text()) for f in sorted((RES / exp).glob("*_summary.json"))]
    return pd.DataFrame(rows)


# S1: size scan
files = sorted((RES / "E5").glob("complete-*__iso.tsv"))
if files:
    d = pd.concat([pd.read_csv(f, sep="\t") for f in files])
    d = d[d.alpha.round(2).isin([0.55, 0.65, 0.75, 0.90])].sort_values(["alpha", "n"])
    rows = [(f"{r.alpha:.2f}", int(r.n), f"{r.F:.6g}", int(r.active), int(r.cycle_rank),
             f"{r.mean_degree:.2f}", int(r.max_degree)) for r in d.itertuples()]
    table("tab:S1", "Size scan on the complete candidate graph (Table IV and Fig.~4 of the main "
          "text): best state found at each exponent and size. $\\overline R$ at unit budget; "
          "$\\bar k$ and $k_{\\max}$ are the mean and maximum degree of the support.",
          ["$\\alpha$", "$n$", "$\\overline R$", "Active", "$\\beta$", "$\\bar k$", "$k_{\\max}$"], rows)

# S2: alpha*(n) on K_n
s = summaries("E1")
B = RES / "E4" / "ansatz_bound.tsv"
if len(s):
    b = pd.read_csv(B, sep="\t") if B.exists() else None
    if b is not None:
        b["best"] = b[["alpha_unif", "alpha_opt"]].min(axis=1)
    rows = []
    for r in s.sort_values("n").itertuples():
        bb = b[b.n == r.n] if b is not None else pd.DataFrame()
        bnd = f"{bb.best.min():.4f}" if len(bb) else "--"
        kb = int(bb.loc[bb.best.idxmin(), "k"]) if len(bb) else "--"
        t = pd.read_csv(RES / "E1" / f"complete-{r.n}__iso.tsv", sep="\t")
        gap = t[t.cycle_rank > 0].blind_gap_rel.max()
        rows.append((int(r.n), f"{r.alpha_star:.3f}", V1.get(int(r.n), "--"), bnd, kb,
                     int(r.active_above), int(r.beta_above), f"{r.mean_degree_above:.2f}",
                     f"{100 * gap:.1f}"))
    table("tab:S2", "Loop-collapse exponent on the complete candidate graph, grid step 0.005. "
          "$\\alpha^{*}$: revised search (unstructured, regular and tree starts with continuation "
          "and alternating passes); $\\alpha^{*}_{\\rm unstr}$: unstructured starts with downward "
          "continuation only; bound: certified $\\min_k\\alpha_k(n)$ with minimizing degree $k$. "
          "Active, $\\beta$, $\\bar k$: looped state one grid step above $\\alpha^{*}$. Last column: "
          "largest relative excess in $\\overline R$ of the best unstructured looped state without "
          "continuation over the revised state, over the looped part of the sweep.",
          ["$n$", "$\\alpha^{*}$", "$\\alpha^{*}_{\\rm unstr}$", "bound", "$k$", "Active", "$\\beta$",
           "$\\bar k$", "excess (\\%)"], rows, star=True)

# S3: bounds
if B.exists():
    b = pd.read_csv(B, sep="\t")
    rows = []
    for n, g in b.groupby("n"):
        cells = [int(n)]
        for k in (3, 4, 5, 6, 7, 8):
            gg = g[g.k == k]
            if len(gg):
                v = gg[["alpha_unif", "alpha_opt"]].min(axis=1).iloc[0]
                cells.append(f"{v:.4f}")
            else:
                cells.append("--")
        cells.append(f"{np.log(n / 2) / np.log(n - 1):.4f}")
        rows.append(cells)
    rows.append(["$\\infty$"] + [f"{np.log(k / 2) / np.log(k * (k - 2) / (k - 1)):.4f}"
                                 for k in (3, 4, 5, 6, 7, 8)] + ["1"])
    table("tab:S3", "Certified upper bounds $\\alpha_k(n)$, Eq.~(17) of the main text, from "
          "low-resistance $k$-regular graphs (conductances optimized on the fixed support up to "
          "$n=128$, uniform beyond), with the star--complete crossing $\\alpha_c(n)$. The last row "
          "is Eq.~(18).",
          ["$n$", "$k=3$", "$4$", "$5$", "$6$", "$7$", "$8$", "$\\alpha_c$"], rows, star=True)


def control_rows(exp):
    s = summaries(exp)
    if not len(s):
        return []
    rows = []
    for r in s.sort_values(["graph", "load", "n"]).itertuples():
        if getattr(r, "all_trees", False) is True:
            a_s = f"$\\ge{r.alpha_star_lower_bound:.2f}$ (no loops)"
        elif pd.isna(r.alpha_star):
            a_s = "--"
        else:
            a_s = f"{r.alpha_star:.3f}"
        above = "--" if pd.isna(getattr(r, "active_above", np.nan)) else \
            f"{int(r.active_above)}/{int(r.beta_above)}"
        rows.append((r.graph.replace("_", "\\_"), r.load.replace("_", "\\_"), int(r.n), int(r.m),
                     a_s, above, f"{r.step:g}"))
    return rows


rows = control_rows("E2")
if rows:
    table("tab:S4", "Collapse exponent on non-complete candidate graphs under isotropic load "
          "(Fig.~8(a) of the main text). Specifications as in the repository "
          "(\\texttt{rrg:D:N:seed}, \\texttt{ba:M:N:seed}, ...). Last columns: active edges and "
          "cycle rank of the looped state one grid step above $\\alpha^{*}$, and the grid step.",
          ["graph", "load", "$n$", "$m$", "$\\alpha^{*}$", "Active/$\\beta$", "step"], rows,
          align="llrrrrr", star=True)
rows = control_rows("E3")
if rows:
    table("tab:S5", "Collapse exponent under Corson's load (\\texttt{corson:SIGMA}, sink at node 0) "
          "and under the mixtures $Q_\\lambda$ (\\texttt{mix:LAMBDA}) (Fig.~8(b) of the main text).",
          ["graph", "load", "$n$", "$m$", "$\\alpha^{*}$", "Active/$\\beta$", "step"], rows,
          align="llrrrrr", star=True)

C = RES / "controls"
if (C / "threshold.tsv").exists():
    t = pd.read_csv(C / "threshold.tsv", sep="\t")
    rows = [(f"{r.alpha:.2f}", int(r.n), f"{r.cut:.0e}", int(r.active), int(r.cycle_rank),
             f"{r.F:.8g}", f"{r.min_share:.2e}") for r in t.itertuples()]
    table("tab:S6", "Truncation-threshold control. The search is run with truncation disabled and "
          "the same pretruncation solution is refined at each cut.",
          ["$\\alpha$", "$n$", "cut", "Active", "$\\beta$", "$\\overline R$", "min share"], rows)

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text("\n".join(out))
print("wrote", a.out, len(out), "tables")
