"""Objective, gradient and multistart optimizer for resistance-optimal networks.

The mathematics and the search protocol are those of the version-1 supplement
(legacy/pre_all.py); this module changes how they are computed, not what is
computed.

Objective. For a candidate graph with edge list E, conductances w >= 0 and a
symmetric positive-semidefinite load matrix Q with Q 1 = 0,

    F(w) = c * tr(L(w)^+ Q),                                         (G1)

with c a positive constant. Isotropic (all-pairs) loading is Q = I - 11^T/n
with c = 2/(n-1), which gives the mean pairwise effective resistance, Eq. (2)
of the manuscript. A single source-sink pair is Q = b_st b_st^T, c = 1.
Corson's fluctuating loads are Q = <i i^T> (see loads.py).

Gradient. dF/dw_e = -c b_e^T L^+ Q L^+ b_e. With M = L^+ Q L^+ and e = (i, j)
this is -c (M_ii + M_jj - 2 M_ij), which costs O(n^3) for all edges at once
instead of the O(n^2 m) of forming L^+ B explicitly. For isotropic loading
M = (L^+)^2 and the gradient is -c d_B^2(e), Eq. (6).

Feasibility. A weighting whose positive support is disconnected has F = +inf.
As in version 1, the optimizer receives a large finite penalty and a barrier
direction there, and a numerically singular Laplacian (second eigenvalue below
1e-13 of the largest) is treated the same way. Keeping this rule unchanged is
what lets the regression tests reproduce the published states.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

PENALTY = 1e300     # must exceed any feasible objective; at alpha ~ 0.05 these reach 1e30 and more
BARRIER = 1e12
LOGIT_BOUNDS = (-25.0, 5.0)
DEFAULT_CUT = 1e-5
DENSE_BFGS_MAX = 600   # largest support refined with dense BFGS


# --------------------------------------------------------------------- problem

@dataclass
class Problem:
    """A candidate graph, a cost law and a load.

    n, edges     candidate graph; edges are (i, j) with i < j, nodes 0..n-1
    alpha        cost exponent in C = sum_e a_e w_e^alpha
    edge_cost    a_e (default 1)
    Q            load matrix (None means isotropic all-pairs loading)
    scale        constant c in (G1); default 2/(n-1) for isotropic loading,
                 1 otherwise
    """
    n: int
    edges: list
    alpha: float
    edge_cost: np.ndarray | None = None
    Q: np.ndarray | None = None
    scale: float | None = None
    _I: np.ndarray = field(init=False, repr=False)
    _J: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.edges = [(int(min(i, j)), int(max(i, j))) for i, j in self.edges]
        self.m = len(self.edges)
        self._I = np.fromiter((e[0] for e in self.edges), int, self.m)
        self._J = np.fromiter((e[1] for e in self.edges), int, self.m)
        self.ec = (np.ones(self.m) if self.edge_cost is None
                   else np.asarray(self.edge_cost, float))
        if self.scale is None:
            self.scale = 2.0 / (self.n - 1) if self.Q is None else 1.0
        self.isotropic = self.Q is None
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        G.add_edges_from(self.edges)
        self.candidate_connected = nx.is_connected(G)

    def sub(self, mask):
        """The same problem restricted to the edges selected by mask."""
        mask = np.asarray(mask, bool)
        return Problem(self.n, [e for e, k in zip(self.edges, mask) if k],
                       self.alpha, self.ec[mask], self.Q, self.scale)

    def with_alpha(self, alpha):
        return Problem(self.n, self.edges, alpha, self.ec, self.Q, self.scale)

    # ------------------------------------------------------------ evaluation

    def support_connected(self, w):
        pos = w > 0
        if pos.all():
            return self.candidate_connected
        A = coo_matrix((np.ones(int(pos.sum())), (self._I[pos], self._J[pos])),
                       shape=(self.n, self.n))
        return connected_components(A, directed=False)[0] == 1

    def laplacian(self, w):
        L = np.zeros((self.n, self.n))
        L[self._I, self._J] = -w
        L[self._J, self._I] = -w
        d = (np.bincount(self._I, w, self.n) + np.bincount(self._J, w, self.n))
        L[np.diag_indices(self.n)] = d
        return L

    def spectral(self, w, rel_tol=1e-13):
        """Eigen-decomposition of L restricted to 1-perp, or None if infeasible."""
        w = np.asarray(w, float)
        if (not np.all(np.isfinite(w))) or np.any(w < 0) or w.max(initial=0.0) > 1e100:
            return None
        if not self.support_connected(w):
            return None
        # The objective is homogeneous, L(c w) = c L(w), so the spectrum is computed
        # for w / max(w) and rescaled. Without this, at small alpha every
        # conductance is far below one (w = p^(1/alpha)), the whole spectrum falls
        # under the singularity floor, and every state is wrongly rejected. With
        # max(w) >= 1 the rule is the version-1 rule unchanged.
        wmax = float(w.max(initial=0.0))
        if wmax <= 0.0:
            return None
        scale = min(wmax, 1.0)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            L = self.laplacian(w / scale)
            if not np.all(np.isfinite(L)):
                return None
            try:
                vals, vecs = np.linalg.eigh(L)
            except np.linalg.LinAlgError:
                return None
            if len(vals) < 2 or not np.all(np.isfinite(vals)):
                return None
            if vals[1] <= rel_tol * max(float(vals[-1]), 1.0):
                return None
        return vals[1:] * scale, vecs[:, 1:]

    def value_grad(self, w):
        """F(w) and dF/dw. Infeasible points return (PENALTY, barrier direction)."""
        sp = self.spectral(w)
        if sp is None:
            return PENALTY, -BARRIER * np.ones_like(w)
        lam, V = sp
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            if self.isotropic:
                value = self.scale * float(np.sum(1.0 / lam))
                M = (V / lam ** 2) @ V.T
            else:
                Lp = (V / lam) @ V.T
                LQ = Lp @ self.Q
                value = self.scale * float(np.trace(LQ))
                M = LQ @ Lp
            dM = np.diag(M)
            g = -self.scale * (dM[self._I] + dM[self._J] - 2.0 * M[self._I, self._J])
        if not (np.isfinite(value) and np.all(np.isfinite(g))):
            return PENALTY, -BARRIER * np.ones_like(w)
        return value, g

    def value(self, w):
        return self.value_grad(w)[0]


# ------------------------------------------------------------ share coordinates

def _unpack(prob, z):
    z = np.clip(z - z.max(), -45.0, 0.0)
    p = np.exp(z)
    p /= p.sum()
    return p, (p / prob.ec) ** (1.0 / prob.alpha)


def _f_logits(prob, z):
    p, w = _unpack(prob, z)
    r, gw = prob.value_grad(w)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        gp = gw * (w / (prob.alpha * np.maximum(p, 1e-300)))
        gz = p * (gp - gp @ p)
    gz = np.where(np.isfinite(gz), gz, 0.0)
    if not np.isfinite(r) or r >= PENALTY:
        return PENALTY, gz
    return r, gz


def refine_support(prob, p_full, cut=DEFAULT_CUT):
    """Set shares at or below cut to zero and re-optimize on the remaining support.

    Identical to _refine_connected_support of version 1 (isotropic loading
    requires a connected support; other loads require only that the support
    reach every node carrying load).
    """
    mask = np.asarray(p_full) > cut
    sub = prob.sub(mask)
    H = nx.Graph()
    H.add_nodes_from(range(prob.n))
    H.add_edges_from(sub.edges)
    if not _load_connected(prob, H):
        return None
    q0 = np.asarray(p_full)[mask].astype(float)
    q0 /= q0.sum()

    def unpack_reduced(y):
        logits = np.r_[y, 0.0]
        logits = np.clip(logits - logits.max(), -60.0, 0.0)
        q = np.exp(logits)
        q /= q.sum()
        return q, (q / sub.ec) ** (1.0 / sub.alpha)

    def f_reduced(y):
        q, w = unpack_reduced(y)
        value, gw = sub.value_grad(w)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            gp = gw * (w / (sub.alpha * q))
            gz = q * (gp - gp @ q)
        gz = np.where(np.isfinite(gz), gz, 0.0)
        if not np.isfinite(value) or value >= PENALTY:
            return PENALTY, gz[:-1]
        return value, gz[:-1]

    if sub.m == 1:
        q = np.ones(1)
        w = (q / sub.ec) ** (1.0 / sub.alpha)
        value, _ = sub.value_grad(w)
        res_x, nit, grad_inf, ok = np.zeros(0), 0, 0.0, np.isfinite(value)
    else:
        y0 = np.log(q0[:-1] / q0[-1])
        # Dense BFGS (version 1) keeps an m x m inverse Hessian. On supports of
        # thousands of edges (dense states near alpha = 1 on large graphs) that is
        # O(m^2) memory and work per iteration and dominates the run, so large
        # supports are refined with limited-memory BFGS instead. Supports of up to
        # DENSE_BFGS_MAX edges, which include every version-1 state, are unchanged.
        if sub.m - 1 <= DENSE_BFGS_MAX:
            res = minimize(f_reduced, y0, jac=True, method="BFGS",
                           options={"maxiter": 3000, "gtol": 1e-8})
        else:
            res = minimize(f_reduced, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": 3000, "gtol": 1e-8, "ftol": 1e-13})
        res_x, nit = res.x, int(res.nit)
        value, gradient = f_reduced(res.x)
        grad_inf = float(np.max(np.abs(gradient), initial=0.0))
        ok = bool(np.isfinite(value) and value < PENALTY and (res.success or grad_inf < 1e-5))
    q, w = unpack_reduced(res_x)
    full_p = np.zeros(prob.m)
    full_w = np.zeros(prob.m)
    full_p[mask] = q
    full_w[mask] = w
    return {"value": float(value), "p": full_p, "w": full_w, "success": bool(ok),
            "projected_grad_inf": grad_inf, "nit": nit}


def _load_connected(prob, H):
    if prob.isotropic:
        return nx.is_connected(H)
    loaded = np.nonzero(np.abs(prob.Q).sum(axis=1) > 0)[0]
    if len(loaded) == 0:
        return True
    comp = nx.node_connected_component(H, int(loaded[0]))
    return all(int(v) in comp for v in loaded)


def descend(prob, z0, cut=DEFAULT_CUT, prune=None):
    """One start: L-BFGS-B in logit coordinates, then support refinement."""
    if prune is None:
        prune = prob.alpha <= 1.0
    lo, hi = LOGIT_BOUNDS
    res = minimize(lambda z: _f_logits(prob, z), z0, jac=True, method="L-BFGS-B",
                   bounds=[(lo, hi)] * prob.m,
                   options={"maxiter": 2500, "ftol": 1e-12, "gtol": 1e-8})
    p, w = _unpack(prob, res.x)
    value, gradient = _f_logits(prob, res.x)
    refined = refine_support(prob, p, cut=cut) if prune else None
    if refined is not None and refined["value"] <= value * (1 + 1e-10):
        return dict(value=refined["value"], p=refined["p"], w=refined["w"],
                    success=refined["success"],
                    projected_grad_inf=refined["projected_grad_inf"],
                    nit=int(res.nit) + refined["nit"])
    projected = gradient.copy()
    projected[(res.x <= lo + 1e-7) & (gradient > 0)] = 0.0
    projected[(res.x >= hi - 1e-7) & (gradient < 0)] = 0.0
    return dict(value=float(value), p=p, w=w,
                success=bool(res.success and np.isfinite(value) and value < PENALTY),
                projected_grad_inf=float(np.max(np.abs(projected))), nit=int(res.nit))


def _descend_job(args):
    prob, z0, cut, prune = args
    return descend(prob, z0, cut, prune)


def summarize_state(prob, p, w, cut=DEFAULT_CUT):
    active_edges = [e for e, share in zip(prob.edges, p) if share > cut]
    S = nx.Graph()
    S.add_nodes_from(range(prob.n))
    S.add_edges_from(active_edges)
    active_nodes = {v for e in active_edges for v in e}
    comps = (nx.number_connected_components(S.subgraph(active_nodes))
             if active_nodes else 0)
    degs = np.array([d for _, d in S.degree()])
    return dict(active=len(active_edges),
                cycle_rank=int(len(active_edges) - len(active_nodes) + comps),
                spans=bool(nx.is_connected(S)),
                eff_support=float(1.0 / np.sum(np.asarray(p) ** 2)),
                max_degree=int(degs.max()) if len(degs) else 0,
                mean_degree=float(degs.mean()) if len(degs) else 0.0,
                budget_residual=float(abs(np.sum(prob.ec * np.asarray(w) ** prob.alpha) - 1.0)))


def multistart(prob, starts, labels=None, cut=DEFAULT_CUT, prune=None, workers=1):
    """Run every start (a list of share vectors) and return the best state.

    Shares are converted to logits exactly as in version 1. With workers > 1
    the starts run in separate processes; the result does not depend on the
    number of workers.
    """
    z_starts = [np.log(np.maximum(np.asarray(s, float) / np.sum(s), 1e-15)) for s in starts]
    labels = list(labels) if labels is not None else [f"start{k}" for k in range(len(starts))]
    jobs = [(prob, z, cut, prune) for z in z_starts]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
            outs = list(ex.map(_descend_job, jobs, chunksize=1))
    else:
        outs = [_descend_job(j) for j in jobs]
    vals = np.array([o["value"] for o in outs])
    if not np.any(np.isfinite(vals) & (vals < PENALTY)):
        raise RuntimeError("No finite state was obtained in the multistart search")
    k = int(np.argmin(vals))
    best = outs[k]
    out = dict(best=float(vals[k]), p=best["p"], w=best["w"], vals=vals,
               winner=labels[k], labels=labels, cut=float(cut),
               n_starts=len(vals), best_success=bool(best["success"]),
               success_fraction=float(np.mean([o["success"] for o in outs])),
               projected_logit_grad_inf=best["projected_grad_inf"],
               best_iterations=best["nit"],
               frac_at_best=float(np.mean(vals <= vals[k] * (1 + 1e-6))),
               spread_rel=float((vals.max() - vals.min()) / vals.min()))
    out.update(summarize_state(prob, best["p"], best["w"], cut))
    return out


def set_single_thread():
    """Pin BLAS to one thread per process; call before spawning workers."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
