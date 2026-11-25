#!/usr/bin/env python
"""Figures of the revised manuscript, from the result folders.

  python scripts/figures_revision.py --res results --out figures

Expected inputs (any missing input skips the figures that need it):
  <res>/E1/complete-N__iso.tsv, _summary.json   collapse sweeps on K_n
  <res>/E4/ansatz_bound.tsv                      certified bounds
  <res>/E5/complete-N__iso.tsv                   size scan at fixed exponents
  <res>/E2/*_summary.json, <res>/E3/*_summary.json   topology and load controls
  <res>/tables/                                  revision_tables.py outputs
The house style (fonts, palette, markers) is the version-1 style of legacy/pre_all.py.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "legacy"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--res", default="results")
ap.add_argument("--out", default="figures")
args = ap.parse_args()
RES = Path(args.res).resolve()
OUT = Path(args.out).resolve()
OUT.mkdir(parents=True, exist_ok=True)
_cwd = os.getcwd()
os.chdir("/tmp")
import pre_all as pa  # noqa: E402  (house style only)
os.chdir(_cwd)
from costconvexity import ansatz, starts  # noqa: E402

BLUE, ORANGE, GREEN, PURPLE = pa.BLUE, pa.ORANGE, pa.GREEN, pa.PURPLE
GREY = pa.GREY
SINGLE, DOUBLE = pa.SINGLE, pa.DOUBLE


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


def summaries(exp):
    rows = []
    for f in sorted((RES / exp).glob("*_summary.json")):
        d = json.loads(f.read_text())
        rows.append(d)
    return pd.DataFrame(rows)


def alpha_c(n):
    return np.log(n / 2) / np.log(n - 1)


# ------------------------------------------------------------------ Fig. 1, 3
def fig1_fig3():
    T = RES / "tables"
    if not (T / "fig1_curve_n12.tsv").exists():
        return
    pa.paper_style()
    n = 12
    cur = pd.read_csv(T / "fig1_curve_n12.tsv", sep="\t")
    aa = np.linspace(0.4, 1.05, 400)
    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
    ax.set(xlim=(0.38, 1.0), ylim=(0, 1.75))
    ax.plot(aa, (n - 1) * (2.0 / n) ** (1.0 / aa), "-", color=BLUE, lw=1.2,
            label=r"$\overline{R}_{\rm star}/\overline{R}_{\rm complete}$")
    pa._refline(ax, 1.0, "equal resistance", x=0.40)
    ac = alpha_c(n)
    ax.axvline(ac, color=ORANGE, ls="--", lw=0.9, zorder=0)
    ax.annotate(rf"$\alpha_c = {ac:.3f}$", (ac, 1.66), xytext=(3, 0),
                textcoords="offset points", fontsize=7, color=ORANGE, va="top")
    ax.plot(cur.alpha, cur.R_best / cur.R_complete, "o", ms=3.6, mec=GREEN, mfc="white",
            mew=0.9, ls="none", zorder=4, label="best state found")
    ax.set(xlabel=r"cost exponent $\alpha$", ylabel="resistance / uniform complete")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "fig1_star_complete.png")

    t3 = pd.read_csv(T / "table3_complete_n12.tsv", sep="\t")
    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
    m_all = n * (n - 1) // 2
    ax.set(xlim=(0.42, 1.0), ylim=(0, m_all * 1.18))
    pa._refline(ax, m_all, f"all {m_all} candidate edges", x=0.44)
    pa._refline(ax, n - 1, "spanning tree (star)", x=0.44, ls=":")
    pa._series(ax, t3.alpha, t3.active, 0, label=r"active edges, $p_e>10^{-5}$")
    pa._series(ax, t3.alpha, t3.eff_support, 1, label=r"effective support $1/\sum_e p_e^2$")
    for _, r in t3.iterrows():
        if 0 < r.cycle_rank and r.active < m_all:
            ax.annotate(rf"$k={int(round(r.mean_degree))}$", (r.alpha, r.active),
                        xytext=(-4, 5), textcoords="offset points", fontsize=6.5,
                        color=pa.LABEL, ha="right")
    ax.set(xlabel=r"cost exponent $\alpha$", ylabel="conductance support")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88))
    fig.tight_layout()
    save(fig, "fig3_support.png")


# ------------------------------------------------------------------ Fig. 2
def hull_points(n=64, ks=(3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32)):
    pts = [("star", n - 1, 2.0 * (n - 1) / n)]
    for j in (1, 2, 4, 8, 16, (n - 1) // 2):
        G = nx.star_graph(n - 1)
        for t in range(j):
            G.add_edge(1 + 2 * t, 2 + 2 * t)
        pts.append(("star+%d" % j, G.number_of_edges(), starts.mean_resistance_unweighted(G, n)))
    for k, (A, H) in sorted(starts.best_regular_graphs(n, ks, samples=10, seed=n).items()):
        pts.append((f"reg{k}", n * k // 2, A))
    pts.append(("complete", n * (n - 1) // 2, 2.0 / n))
    return pd.DataFrame(pts, columns=["label", "m", "A"])


def fig2():
    s = summaries("E1")
    B = RES / "E4" / "ansatz_bound.tsv"
    if not len(s) or not B.exists():
        return
    pa.paper_style()
    fig, axs = plt.subplots(2, 2, figsize=(DOUBLE, 5.0))
    ax = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]

    a = ax[0]
    b = pd.read_csv(B, sep="\t")
    b["best"] = b[["alpha_unif", "alpha_opt"]].min(axis=1)
    g = b.groupby("n")["best"].min()
    ns = np.geomspace(8, g.index.max(), 200)
    a.plot(ns, alpha_c(ns), "-", color=GREY, lw=1.0, label=r"$\alpha_c(n)$, star vs complete")
    a.plot(g.index, g.values, "-", color=BLUE, lw=1.2, label="certified bound (regular)")
    a.axhline(ansatz.km_limit(5), color=BLUE, ls=":", lw=0.8)
    a.annotate(r"$\alpha_5(\infty)=0.693$", (g.index.max(), ansatz.km_limit(5)),
               xytext=(0, -9), textcoords="offset points", ha="right", fontsize=6.5, color=BLUE)
    s = s[s.alpha_star.notna()].sort_values("n")
    a.errorbar(s.n, s.alpha_star, yerr=[np.zeros(len(s)), np.full(len(s), 0.005)], fmt="s",
               ms=3.4, mec=ORANGE, mfc="white", mew=0.9, ecolor=ORANGE, elinewidth=0.7,
               capsize=1.5, ls="none", label=r"$\alpha^{*}(n)$, search")
    a.set(xscale="log", xlabel=r"$n$", ylabel=r"cost exponent $\alpha$", ylim=(0.54, 0.93))
    ticks = [8, 16, 32, 64, 128, 256, 512, 2048]
    a.set_xticks([t for t in ticks if t <= g.index.max()])
    a.set_xticklabels([str(t) for t in ticks if t <= g.index.max()])
    a.minorticks_off()
    a.text(0.97, 0.03, "trees", transform=a.transAxes, ha="right", fontsize=7, color=ORANGE)
    a.text(0.55, 0.60, "loops", transform=a.transAxes, ha="left", fontsize=7, color=BLUE)
    a.legend(loc="upper left", fontsize=6.5)
    pa._panel_tag(a, "a", dx=-0.17)

    c = ax[1]
    f64 = RES / "E1" / "complete-64__iso.tsv"
    if f64.exists():
        t = pd.read_csv(f64, sep="\t").sort_values("alpha")
        rs = np.array([ansatz.star_R(64, x) for x in t.alpha])
        k = (2 * t.active / 64).round().astype(int)
        for kk, mk, col in ((1, "o", ORANGE), (4, "s", BLUE), (5, "^", GREEN)):
            sel = (k == kk) if kk > 1 else (t.cycle_rank == 0)
            lab = "star" if kk == 1 else f"{kk}-regular"
            c.plot(t.alpha[sel], (t.F / rs)[sel], mk, ms=3.2, mec=col, mfc="white", mew=0.9,
                   ls="none", label=lab)
        c.axhline(1.0, color=GREY, lw=0.6, zorder=0)
        astar = float(t[t.cycle_rank == 0].alpha.max())
        c.axvline(astar + 0.0025, color=GREY, ls=":", lw=0.6)
        c.set(xlabel=r"cost exponent $\alpha$", ylabel=r"$\overline{R}_{\rm best}/\overline{R}_{\rm star}$")
        c.legend(loc="lower left", fontsize=6.5)
        c.set_title(r"$n=64$", fontsize=8)
    pa._panel_tag(c, "b")

    d = ax[2]
    hp = hull_points(64)
    hp.to_csv(OUT / "fig2c_hull_points_n64.tsv", sep="\t", index=False)
    st = hp[hp.label == "star"].iloc[0]
    r4 = hp[hp.label == "reg4"].iloc[0]
    a4 = np.log(r4.m / st.m) / np.log(st.A / r4.A)     # facet slope -1/a4
    # Objective of a uniform state relative to the star at alpha = a4:
    # log[A m^(1/alpha)] - log[A_star (n-1)^(1/alpha)]. The star and the
    # 4-regular graph tie at zero; every other graph lies above.
    hp["dF"] = np.log(hp.A / st.A) + np.log(hp.m / st.m) / a4
    reg = hp[hp.label.str.startswith("reg") | hp.label.isin(["star", "complete"])]
    add = hp[hp.label.str.startswith("star+")]
    d.axhline(0.0, color=GREY, lw=0.6, zorder=0)
    d.plot(add.m, add.dF, "^", ms=3.2, mec=PURPLE, mfc="white", mew=0.9, ls="none",
           label="star + leaf edges")
    d.plot(reg.m, reg.dF, "o", ms=3.4, mec=BLUE, mfc="white", mew=0.9, ls="none",
           label="$k$-regular")
    d.plot([st.m], [0.0], "o", ms=4.2, mec=ORANGE, mfc=ORANGE, ls="none", label="star")
    for _, r in reg.iterrows():
        if r.label.startswith("reg") and r.label in ("reg3", "reg4", "reg5", "reg6", "reg8", "reg16"):
            d.annotate(r.label.replace("reg", "$k=") + "$", (r.m, r.dF), xytext=(0, 4),
                       textcoords="offset points", fontsize=6, color=pa.LABEL, ha="center")
    d.set(xscale="log", xlabel=r"edges $m_H$", ylim=(-0.05, 0.6), xlim=(50, 1200),
          ylabel=r"$\Delta\log\overline{R}$ at $\alpha=%.3f$" % a4)
    d.set_title(r"$n=64$, uniform conductance", fontsize=8)
    d.legend(loc="upper left", fontsize=6.0)
    pa._panel_tag(d, "c")
    phase_panel(ax[3])
    fig.tight_layout()
    save(fig, "fig2_collapse.png")


def phase_panel(e):
    """(d) Degree of the best state in the (alpha, n) plane: uniform-weight
    prediction as background bands, search results (E6, else E5) as symbols."""
    from matplotlib.colors import BoundaryNorm, ListedColormap
    P = RES / "E4" / "phase_prediction.tsv"
    levels = [1, 3, 4, 5, 6, 8, 10, 12, 16, 24, 1e9]
    names = ["star", "3", "4", "5", "6–7", "8–9", "10–11", "12–15", "16–23", "≥24"]
    cols = ["#E8DCCB", "#D7E3F0", "#B9CDE5", "#9AB5D8", "#7C9CC9", "#6083B6", "#486A9E",
            "#335284", "#233C69", "#14264A"]
    cmap = ListedColormap(cols)
    norm = BoundaryNorm(levels, cmap.N)
    if P.exists():
        p = pd.read_csv(P, sep="\t")
        ns = np.sort(p.n.unique())
        edges_n = np.r_[ns[0] / 1.15, np.sqrt(ns[:-1] * ns[1:]), ns[-1] * 1.15]
        al = np.sort(p.alpha.unique())
        da = al[1] - al[0]
        Z = np.full((len(ns), len(al)), np.nan)
        for i, n in enumerate(ns):
            g = p[p.n == n].sort_values("alpha")
            Z[i] = g.k.values
        e.pcolormesh(np.r_[al - da / 2, al[-1] + da / 2], edges_n, Z, cmap=cmap, norm=norm,
                     shading="flat", rasterized=True)
    src = "E6" if any((RES / "E6").glob("*.tsv")) else "E5"
    files = sorted((RES / src).glob("complete-*__iso.tsv"))
    if files:
        d = pd.concat([pd.read_csv(f, sep="\t") for f in files])
        k = np.where(d.cycle_rank == 0, 1, 2 * d.active / d.n)
        e.scatter(d.alpha, d.n, c=k, cmap=cmap, norm=norm, s=9, edgecolors="0.15",
                  linewidths=0.4, zorder=3)
    e.set(yscale="log", xlabel=r"cost exponent $\alpha$", ylabel=r"$n$", xlim=(0.5, 0.99))
    yt = [8, 16, 32, 64, 128, 256, 512]
    e.set_yticks(yt)
    e.set_yticklabels([str(t) for t in yt])
    e.minorticks_off()
    from matplotlib.patches import Patch
    e.legend(handles=[Patch(color=c, label=l) for c, l in zip(cols, names)], title="degree",
             fontsize=5.5, title_fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5),
             frameon=False)
    pa._panel_tag(e, "d")


# ------------------------------------------------------------------ Fig. 4
def kstar(alpha):
    ks = np.arange(3, 200)
    f = np.log(2 * (ks - 1) / (ks * (ks - 2))) + np.log(ks) / alpha
    return int(ks[np.argmin(f)])


_REG_CACHE = {}


def kstar_n(alpha, n, ks=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16)):
    """Degree minimizing A_k(n) (kn/2)^(1/alpha) over low-resistance k-regular graphs,
    or 0 if the star wins (then nothing is drawn)."""
    if n not in _REG_CACHE:
        _REG_CACHE[n] = starts.best_regular_graphs(n, ks, samples=4, seed=n)
    vals = {k: np.log(A) + np.log(n * k / 2) / alpha for k, (A, _) in _REG_CACHE[n].items()}
    star = np.log(2 * (n - 1) / n) + np.log(n - 1) / alpha
    k = min(vals, key=vals.get)
    return k if vals[k] < star else 2 * (n - 1) / n


def fig4():
    files = sorted((RES / "E5").glob("complete-*__iso.tsv"))
    if not files:
        return
    df = pd.concat([pd.read_csv(f, sep="\t") for f in files])
    df.to_csv(OUT / "tableS1_size_scan.tsv", sep="\t", index=False)
    pa.paper_style()
    fig, ax = plt.subplots(1, 3, figsize=(DOUBLE, 2.4))
    alphas = (0.55, 0.65, 0.75, 0.90)
    nmax = df.n.max()
    ns = np.array(sorted(df.n.unique()))
    ax[0].plot(ns, ns - 1, ":", color=GREY, lw=0.7)
    ax[0].plot(ns, ns * (ns - 1) / 2, "-", color=GREY, lw=0.7)
    for k, a in enumerate(alphas):
        g = df[np.isclose(df.alpha, a)].sort_values("n")
        pa._series(ax[0], g.n, g.active, k, label=rf"$\alpha={a}$")
        pa._series(ax[1], g.n, g.active / (g.n - 1), k)
        pa._series(ax[2], g.n, g.cycle_rank / g.n, k)
        if a > 0.6:
            pred = [kstar_n(a, int(nn)) for nn in g.n]
            ax[1].plot(g.n, np.array(pred) / 2 * g.n / (g.n - 1), "_", ms=9, mew=1.4,
                       color=pa.CYCLE[k], alpha=0.55, zorder=1)
    ax[0].set(xscale="log", yscale="log", xlabel=r"$n$", ylabel="active edges")
    ax[0].legend(loc="upper left", fontsize=6.5)
    ax[1].set(xscale="log", xlabel=r"$n$", ylabel=r"active edges $/(n-1)$")
    ax[2].set(xscale="log", xlabel=r"$n$", ylabel=r"cycle rank $\beta/n$")
    for i, x in enumerate(ax):
        x.set_xticks([t for t in (8, 16, 32, 64, 128) if t <= nmax])
        x.set_xticklabels([str(t) for t in (8, 16, 32, 64, 128) if t <= nmax])
        x.minorticks_off()
        pa._panel_tag(x, "abc"[i])
    fig.tight_layout()
    save(fig, "fig4_scaling.png")


# ------------------------------------------------------------------ Figs. 5, 6
def fig5_6():
    T = RES / "tables"
    if not (T / "spatial_states.npz").exists():
        return
    z = np.load(T / "spatial_states.npz")
    df = pd.read_csv(T / "table5_spatial.tsv", sep="\t")
    pos = z["pos"]
    edges = [tuple(e) for e in z["edges"]]
    G = nx.Graph()
    G.add_nodes_from(range(len(pos)))
    G.add_edges_from(edges)
    payload = {}
    for _, r in df.iterrows():
        key = f"{r.alpha:g}|{r.a_exponent:g}"
        payload[key] = {"p": z[f"{r.alpha:g}_{r.a_exponent:g}_p"], "w": z[f"{r.alpha:g}_{r.a_exponent:g}_w"]}
    old = pa.OUT
    pa.OUT = OUT
    pa.fig5_6_spatial((G, pos, edges, z["lengths"], df, payload))
    pa.OUT = old


# ------------------------------------------------------------------ Fig. 8
def fig8():
    s2 = summaries("E2")
    s1 = summaries("E1")
    s3 = summaries("E3")
    if not len(s2):
        return
    pa.paper_style()
    fig, ax = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))
    a = ax[0]
    s1 = s1[s1.alpha_star.notna()].sort_values("n")
    a.plot(s1.n, s1.alpha_star, "-o", color=BLUE, ms=3, mfc="white", label="complete")
    s2 = s2.copy()
    s2["fam"] = s2.graph.str.split(":").str[0]
    s2.loc[s2.fam.isin(["rrg", "er", "ba"]), "fam"] = s2.graph.str.split(":").str[:2].str.join(":")
    names = {"square": "square lattice", "triangular": "triangular lattice", "rrg:4": "random 4-regular",
             "rrg:8": "random 8-regular", "er:8": "Erdős–Rényi, $c=8$", "ba:3": "Barabási–Albert",
             "rgg": "random geometric"}
    marks = ["s", "^", "D", "v", "P", "X", "*"]
    for k, (fam, g) in enumerate(sorted(s2.groupby("fam"), key=lambda x: x[0])):
        g = g[g.alpha_star.notna()].sort_values("n")
        if len(g):
            a.plot(g.n, g.alpha_star, marker=marks[k % 7], ls="--", lw=0.8, ms=3, mfc="white",
                   color=pa.CYCLE[(k + 1) % 4], label=names.get(fam, fam))
    if len(s3):
        cs = s3[(s3.load == "corson:1") & s3.graph.str.startswith("square")]
        cs = cs[cs.alpha_star.notna()].sort_values("n")
        if len(cs):
            a.plot(cs.n, cs.alpha_star, marker="s", ls="-", lw=0.8, ms=3, mfc="0.3",
                   color="0.3", label="square lattice, Corson load")
    a.set(xscale="log", xlabel=r"$n$", ylabel=r"collapse exponent $\alpha^{*}$", ylim=(0, 0.8))
    nt = [t for t in (8, 16, 32, 64, 128, 256) if t <= max(s1.n.max(), s2.n.max())]
    a.set_xticks(nt)
    a.set_xticklabels([str(t) for t in nt])
    a.minorticks_off()
    a.legend(loc="lower left", fontsize=6, ncol=2)
    pa._panel_tag(a, "a")

    b = ax[1]
    if len(s3):
        s3 = s3.copy()
        s3["lam"] = s3.load.str.extract(r"mix:([0-9.]+)").astype(float)
        s3["fam"] = s3.graph.str.split(":").str[0]
        iso = pd.concat([s1.assign(fam="complete"), s2[s2.fam == "square"]])
        iso = iso.assign(lam=1.0)
        allp = pd.concat([s3, iso], ignore_index=True)
        top = 1.02
        allp["a"] = allp.alpha_star
        if "all_trees" in allp:
            allp.loc[allp.all_trees.fillna(False).astype(bool), "a"] = top
        k = 0
        for (fam, n), g in allp.groupby(["fam", "n"]):
            g = g[g.lam.notna() & g.a.notna()].sort_values("lam")
            if len(g) < 2:
                continue
            b.plot(g.lam, g.a, marker=pa.MARKERS[k % 4],
                   ls="-" if fam == "complete" else "--", lw=0.8, ms=3.4, mfc="white",
                   color=pa.CYCLE[k % 4], label=f"{fam}, $n={n}$")
            k += 1

        b.set(xlabel=r"isotropic fraction $\lambda$ of the load",
              ylabel=r"collapse exponent $\alpha^{*}$", ylim=(0.5, 0.82))
        b.legend(loc="lower left", fontsize=6, ncol=2)
    pa._panel_tag(b, "b")
    fig.tight_layout()
    save(fig, "fig8_control.png")


if __name__ == "__main__":
    fig1_fig3()
    fig2()
    fig4()
    fig5_6()
    fig8()
