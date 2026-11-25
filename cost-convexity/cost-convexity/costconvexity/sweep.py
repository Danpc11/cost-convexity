"""Loop-collapse exponent alpha*(n) by a downward sweep with continuation.

At every exponent of a descending grid the search runs

  * the version-1 start set (uniform, two spanning trees, hub star and
    n_random Dirichlet allocations; same seed at every exponent),
  * the mechanism-informed looped starts of starts.looped_starts, and
  * the best state of the previous (larger) exponent, the continuation start,

and keeps the lowest objective. The sweep stops `confirm` grid steps after the
first spanning tree. alpha* is the largest exponent whose best state is a
spanning tree. Every visited state is written out, with its shares, so the
structure of the states on either side of the collapse can be inspected.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import core, graphs, loads, starts


NEW_PREFIXES = ("reg", "sub", "bfs")


def make_problem(graph_spec, load_spec, alpha):
    n, edges, cost, _ = graphs.build(graph_spec)
    Q, scale = loads.build(load_spec, n)
    return core.Problem(n, edges, alpha, cost, Q, scale)


def sweep(graph_spec, load_spec="iso", amax=0.75, amin=0.30, step=0.005,
          n_random=24, seed=7, confirm=4, looped=True, looped_samples=40,
          workers=1, outdir="results", tag=None, verbose=True, upward=True,
          max_passes=6):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = tag or f"{graph_spec}__{load_spec}".replace(":", "-")
    prob0 = make_problem(graph_spec, load_spec, amax)
    n, edges = prob0.n, prob0.edges

    if looped:
        lstarts, llabels = starts.looped_starts(n, edges, samples=looped_samples, seed=seed + 12345)
        ts, tl = starts.tree_starts(n, edges, seed=seed + 54321)
        lstarts, llabels = lstarts + ts, llabels + tl
    else:
        lstarts, llabels = [], []

    grid = np.round(np.arange(amax, amin - 1e-9, -step), 6)
    rows, shares = [], {}
    carry = None
    trees_seen = 0
    t0 = time.time()
    for a in grid:
        prob = prob0.with_alpha(float(a))
        rng = np.random.default_rng(seed)
        s, lab = starts.legacy_starts(n, edges, rng, n_random)
        s, lab = s + lstarts, lab + llabels
        if carry is not None:
            s.append(carry)
            lab.append("continuation")
        r = core.multistart(prob, s, lab, workers=workers)
        legacy_mask = np.array([not (l.startswith(NEW_PREFIXES) or l == "continuation")
                                for l in lab])
        blind_best = float(np.min(r["vals"][legacy_mask]))
        rows.append(dict(graph=graph_spec, load=load_spec, n=n, m=len(edges),
                         alpha=float(a), F=r["best"], active=r["active"],
                         cycle_rank=r["cycle_rank"], spans=r["spans"],
                         eff_support=r["eff_support"], max_degree=r["max_degree"],
                         mean_degree=r["mean_degree"], winner=r["winner"],
                         F_blind=blind_best,
                         blind_gap_rel=(blind_best - r["best"]) / r["best"],
                         frac_at_best=r["frac_at_best"],
                         grad_inf=r["projected_logit_grad_inf"],
                         budget_residual=r["budget_residual"],
                         n_starts=r["n_starts"], seconds=time.time() - t0))
        shares[f"{a:.4f}"] = r["p"]
        if verbose:
            print(f"[{tag}] alpha={a:.3f} F={r['best']:.8g} active={r['active']} "
                  f"beta={r['cycle_rank']} winner={r['winner']} "
                  f"blind_gap={rows[-1]['blind_gap_rel']:.2e} t={time.time()-t0:.0f}s",
                  flush=True)
        carry = r["p"]
        trees_seen = trees_seen + 1 if r["cycle_rank"] == 0 else 0
        if trees_seen > confirm:
            break
    if upward and len(rows) > 1:
        # Alternating continuation passes. Each pass carries the current best
        # state one grid step at a time (upward, then downward, ...) and keeps
        # it wherever it wins. On candidate graphs whose optimal tree is not
        # known in closed form this keeps the tree side of the comparison as
        # strong as the looped side. Passes stop when one changes nothing.
        order = list(range(len(rows)))       # rows are stored in descending alpha
        for pass_no in range(max_passes):
            seq = order[::-1] if pass_no % 2 == 0 else order   # upward first
            changed = 0
            carry = shares[f"{rows[seq[0]]['alpha']:.4f}"]
            for k in seq[1:]:
                a = rows[k]["alpha"]
                r = core.multistart(prob0.with_alpha(a), [carry], ["pass"], workers=1)
                if r["best"] < rows[k]["F"] * (1 - 1e-10):
                    changed += 1
                    if verbose:
                        print(f"[{tag}] pass {pass_no + 1} improves alpha={a:.3f}: "
                              f"{rows[k]['F']:.8g} -> {r['best']:.8g} "
                              f"(beta {rows[k]['cycle_rank']} -> {r['cycle_rank']})", flush=True)
                    rows[k].update(F=r["best"], active=r["active"], cycle_rank=r["cycle_rank"],
                                   spans=r["spans"], eff_support=r["eff_support"],
                                   max_degree=r["max_degree"], mean_degree=r["mean_degree"],
                                   winner=f"pass{pass_no + 1}",
                                   blind_gap_rel=(rows[k]["F_blind"] - r["best"]) / r["best"],
                                   grad_inf=r["projected_logit_grad_inf"],
                                   budget_residual=r["budget_residual"])
                    shares[f"{a:.4f}"] = r["p"]
                carry = shares[f"{a:.4f}"]
            if verbose:
                print(f"[{tag}] pass {pass_no + 1}: {changed} exponents improved", flush=True)
            if changed == 0:
                break
    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"{tag}.tsv", sep="\t", index=False)
    np.savez_compressed(outdir / f"{tag}_shares.npz", edges=np.array(edges), **shares)
    summ = summarize(df)
    summ.update(graph=graph_spec, load=load_spec, n=n, m=len(edges), step=step,
                n_random=n_random, seed=seed, looped=looped, upward=upward)
    (outdir / f"{tag}_summary.json").write_text(json.dumps(summ, indent=1))
    if verbose:
        print(f"[{tag}] summary {summ}", flush=True)
    return df, summ


def summarize(df):
    g = df.sort_values("alpha")
    tree = g[g.cycle_rank == 0]
    looped = g[g.cycle_rank > 0]
    if len(looped) == 0:
        # Every visited state is a tree: alpha* is at or above the top of the grid.
        return dict(alpha_star=None, all_trees=True, bracketed=False,
                    alpha_star_lower_bound=float(g.alpha.max()),
                    alpha_min_visited=float(g.alpha.min()))
    if len(tree) == 0:
        # No tree reached: alpha* lies below the bottom of the grid.
        return dict(alpha_star=None, all_trees=False, bracketed=False,
                    alpha_star_upper_bound=float(g.alpha.min()),
                    alpha_min_visited=float(g.alpha.min()))
    a_star = float(tree.alpha.max())
    above = looped[looped.alpha > a_star]
    first = above.iloc[0] if len(above) else None
    reentrant = bool((looped.alpha < a_star).any())
    return dict(alpha_star=a_star, bracketed=first is not None,
                alpha_first_looped=None if first is None else float(first.alpha),
                active_above=None if first is None else int(first.active),
                beta_above=None if first is None else int(first.cycle_rank),
                mean_degree_above=None if first is None else float(first.mean_degree),
                winner_above=None if first is None else str(first.winner),
                reentrant_loops_below=reentrant,
                tree_steps_below=int((g.alpha <= a_star).sum()))
