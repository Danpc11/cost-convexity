#!/usr/bin/env python
"""Tables III, V and VI of the manuscript with the revised search protocol.

Each table is a short ladder of exponents. At every exponent the search runs the
version-1 start set (same seeds as version 1), the looped and breadth-first
tree starts of starts.py, and the best state of the neighbouring exponent;
alternating continuation passes along the ladder follow until nothing changes.
The version-1 value (version-1 start set alone) is reported alongside.

  python scripts/revision_tables.py --out results/tables
"""
import argparse
import os
import sys
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "legacy"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from costconvexity import core, starts, ansatz  # noqa: E402

NEW = ("reg", "sub", "bfs")


def ladder(problems, n_random, seed, workers=1, max_passes=6):
    """problems: list of (key, Problem) in the order the ladder is walked."""
    out = {}
    carry = None
    for key, prob in problems:
        rng = np.random.default_rng(seed)
        s, lab = starts.legacy_starts(prob.n, prob.edges, rng, n_random)
        ls, ll = starts.looped_starts(prob.n, prob.edges, seed=seed + 12345)
        ts, tl = starts.tree_starts(prob.n, prob.edges, seed=seed + 54321)
        s, lab = s + ls + ts, lab + ll + tl
        if carry is not None:
            s.append(carry)
            lab.append("continuation")
        r = core.multistart(prob, s, lab, workers=workers)
        legacy = np.array([not (l.startswith(NEW) or l == "continuation") for l in lab])
        r["F_v1"] = float(np.min(r["vals"][legacy]))
        r["frac_at_best_v1"] = float(np.mean(r["vals"][legacy] <= r["F_v1"] * (1 + 1e-6)))
        r["spread_v1"] = float((r["vals"][legacy].max() - r["F_v1"]) / r["F_v1"])
        out[key] = r
        carry = r["p"]
    keys = [k for k, _ in problems]
    probs = dict(problems)
    for pass_no in range(max_passes):
        order = keys[::-1] if pass_no % 2 == 0 else keys
        changed = 0
        carry = out[order[0]]["p"]
        for k in order[1:]:
            r = core.multistart(probs[k], [carry], ["pass"])
            if r["best"] < out[k]["best"] * (1 - 1e-10):
                for f in ("F_v1", "frac_at_best_v1", "spread_v1"):
                    r[f] = out[k][f]
                r["frac_at_best"] = out[k]["frac_at_best"]
                r["spread_rel"] = out[k]["spread_rel"]
                r["winner"] = f"pass{pass_no + 1}"
                out[k] = r
                changed += 1
            carry = out[k]["p"]
        if changed == 0:
            break
    return out


def table_complete(out, n=12, alphas=(0.97, 0.90, 0.82, 0.75, 0.65, 0.55, 0.45)):
    E = [(i, j) for i in range(n) for j in range(i + 1, n)]
    res = ladder([(a, core.Problem(n, E, a)) for a in alphas], n_random=60, seed=11)
    rows = []
    for a in sorted(alphas):
        r = res[a]
        rc = 2.0 / (n * (1.0 / len(E)) ** (1.0 / a))
        rows.append(dict(alpha=a, R_best=r["best"], R_complete=rc, R_star=ansatz.star_R(n, a),
                         ratio=r["best"] / rc, ratio_v1=r["F_v1"] / rc, active=r["active"],
                         cycle_rank=r["cycle_rank"], eff_support=r["eff_support"],
                         mean_degree=r["mean_degree"], max_degree=r["max_degree"],
                         winner=r["winner"], frac_at_best=r["frac_at_best"],
                         spread=r["spread_rel"], n_starts=r["n_starts"],
                         frac_at_best_v1=r["frac_at_best_v1"],
                         spread_v1=r["spread_v1"], grad_inf=r["projected_logit_grad_inf"],
                         budget_residual=r["budget_residual"]))
        print(f"K{n} alpha={a}: ratio {rows[-1]['ratio']:.4f} (v1 {rows[-1]['ratio_v1']:.4f}) "
              f"active={r['active']} beta={r['cycle_rank']} winner={r['winner']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "table3_complete_n12.tsv", sep="\t", index=False)
    np.savez_compressed(out / "table3_shares.npz", **{f"{a:.2f}": res[a]["p"] for a in alphas})
    # Fig. 1 curve: best states on a finer grid at n = 12
    fine = tuple(np.round(np.arange(0.95, 0.399, -0.025), 3))
    resf = ladder([(a, core.Problem(n, E, a)) for a in fine], n_random=24, seed=11)
    pd.DataFrame([dict(alpha=a, R_best=resf[a]["best"], active=resf[a]["active"],
                       cycle_rank=resf[a]["cycle_rank"],
                       R_complete=2.0 / (n * (1.0 / len(E)) ** (1.0 / a)),
                       R_star=ansatz.star_R(n, a)) for a in sorted(fine)]
                 ).to_csv(out / "fig1_curve_n12.tsv", sep="\t", index=False)
    return df


def table_spatial(out):
    import pre_all as pa
    G, pos, edges, lengths = pa.spatial_graph()
    n = len(G)
    cases = sorted(pa.SPATIAL_CASES, key=lambda c: -c[0])
    probs = [((a, lex), core.Problem(n, edges, a, lengths ** lex)) for a, lex, _ in cases]
    res_c = ladder([p for p in probs if p[0][0] < 1], n_random=60, seed=41)
    res_v = ladder([p for p in probs if p[0][0] >= 1], n_random=12, seed=41)
    res = {**res_c, **res_v}
    rows = []
    for (a, lex), prob in sorted(probs, key=lambda p: p[0][0]):
        r = res[(a, lex)]
        p_eq = np.full(prob.m, 1.0 / prob.m)
        w_eq = (p_eq / prob.ec) ** (1.0 / a)
        r_eq = prob.value(w_eq)
        q_w = r["w"] / r["w"].sum()
        rows.append(dict(alpha=a, a_exponent=lex, R_best=r["best"], R_v1=r["F_v1"],
                         active=r["active"], cycle_rank=r["cycle_rank"],
                         eff_support=r["eff_support"], eff_support_w=float(1 / np.sum(q_w ** 2)),
                         w_max=float(r["w"].max()), access_gain=r_eq / r["best"],
                         priced_out=prob.m - r["active"], winner=r["winner"]))
        print(f"spatial alpha={a} lex={lex}: active={r['active']} beta={r['cycle_rank']} "
              f"gain={rows[-1]['access_gain']:.2f} R={r['best']:.6g} (v1 {r['F_v1']:.6g}) "
              f"winner={r['winner']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "table5_spatial.tsv", sep="\t", index=False)
    np.savez_compressed(out / "spatial_states.npz", pos=pos, edges=np.array(edges), lengths=lengths,
                        **{f"{a:g}_{lex:g}_p": res[(a, lex)]["p"] for a, lex, _ in cases},
                        **{f"{a:g}_{lex:g}_w": res[(a, lex)]["w"] for a, lex, _ in cases})
    return df


def table_loading(out):
    import pre_all as pa
    G, pos, edges, lengths = pa.spatial_graph(seed=5, n=20, radius=0.42)
    n = len(G)
    hops = dict(nx.all_pairs_shortest_path_length(G))
    s, t = max(((a, b) for a in range(n) for b in range(a + 1, n)), key=lambda p: hops[p[0]][p[1]])
    alphas = (0.97, 0.90, 0.75, 0.65, 0.55, 0.45)
    res = ladder([(a, core.Problem(n, edges, a)) for a in alphas], n_random=60, seed=3)
    rows = []
    for a in sorted(alphas):
        r = res[a]
        path = pa.exact_single_pair_path(G, edges, a, s, t)
        rows.append(dict(alpha=a, eff_allpairs=r["eff_support"], active_allpairs=r["active"],
                         cycles_allpairs=r["cycle_rank"], R_allpairs=r["best"], R_v1=r["F_v1"],
                         path_edges=path["active"], cycles_path=path["cycle_rank"],
                         winner=r["winner"]))
        print(f"loading alpha={a}: all-pairs active={r['active']} beta={r['cycle_rank']} "
              f"R={r['best']:.6g} (v1 {r['F_v1']:.6g}); path {path['active']} edges", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "table6_loading.tsv", sep="\t", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/tables")
    ap.add_argument("--only", default="complete,spatial,loading")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)  # the legacy module writes pre_results/ in the cwd
    out = Path(".")
    for name in a.only.split(","):
        {"complete": table_complete, "spatial": table_spatial, "loading": table_loading}[name](out)
