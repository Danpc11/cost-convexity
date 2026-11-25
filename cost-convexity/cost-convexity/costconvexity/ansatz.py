"""Analytic and semi-analytic results for the complete candidate graph.

Tree lemma. For a spanning tree T with conductances w, the effective resistance
between two nodes is the sum of 1/w_e along the unique path, so the mean
pairwise resistance is

    R(T, w) = [2 / (n (n-1))] sum_e c_e / w_e,   c_e = s_e (n - s_e),

with s_e the number of nodes on one side of e. Minimizing under
sum_e a_e w_e^alpha = 1 (Lagrange) gives w_e^(alpha+1) proportional to c_e / a_e and

    R*(T) = [2 / (n (n-1))] [ sum_e (a_e^(1/alpha) c_e)^q ]^(1/q),  q = alpha/(alpha+1).

For a_e = 1 every c_e >= n - 1, with equality only for a leaf edge, and every
tree has n - 1 edges, so the star (all edges leaf edges) is the unique optimal
spanning tree of K_n at every alpha > 0, with R*_star = 2 (n-1)^(1+1/alpha) / n.

Uniform-weight competitors. A graph H with m_H edges and unit-conductance mean
resistance A(H), given uniform conductance at unit budget, has
R = A(H) m_H^(1/alpha). It beats the star when alpha exceeds

    alpha_H(n) = log(m_H / (n-1)) / log(A_star / A(H)),  A_star = 2 (n-1)/n.

Any such H is an explicit feasible state, so alpha*(n) <= alpha_H(n) is a
certified upper bound on the exponent below which the global optimum can be a
tree. For random k-regular graphs A(H) -> 2(k-1)/(k(k-2)) as n grows (the
Kesten-McKay / regular-tree heuristic, see Johnson and Walters,
arXiv:2206.08095), which gives the limits alpha_k(inf) = log(k/2) /
log(k(k-2)/(k-1)): 1 for k = 3, 0.7067 for k = 4, 0.6932 for k = 5,
0.7004 for k = 6.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from . import core
from .starts import mean_resistance_unweighted


def star_R(n, alpha):
    return 2.0 * (n - 1) ** (1.0 + 1.0 / alpha) / n


def tree_optimal_R(T, alpha, edge_cost=None):
    """Closed-form optimum of the all-pairs objective on a fixed spanning tree."""
    n = T.number_of_nodes()
    edges = list(T.edges())
    a = np.ones(len(edges)) if edge_cost is None else np.asarray(edge_cost, float)
    c = []
    for u, v in edges:
        H = T.copy()
        H.remove_edge(u, v)
        s = len(nx.node_connected_component(H, u))
        c.append(s * (n - s))
    c = np.array(c, float)
    q = alpha / (alpha + 1.0)
    return 2.0 / (n * (n - 1)) * np.sum((a ** (1.0 / alpha) * c) ** q) ** (1.0 / q)


def crossing_alpha(n, A_H, m_H):
    A_star = 2.0 * (n - 1) / n
    if A_H >= A_star:
        return np.inf
    return np.log(m_H / (n - 1)) / np.log(A_star / A_H)


def km_limit(k):
    A = 2.0 * (k - 1) / (k * (k - 2))
    return np.log(k / 2.0) / np.log(2.0 / A)


def regular_bound(n, ks=(3, 4, 5, 6, 7, 8), samples=40, seed=0):
    """Certified upper bounds alpha_k(n) from the best sampled k-regular graph."""
    from .starts import best_regular_graphs
    rows = []
    for k, (A, H) in sorted(best_regular_graphs(n, ks, samples, seed).items()):
        m = n * k // 2
        rows.append(dict(n=n, k=k, A=A, m=m, alpha_k=crossing_alpha(n, A, m),
                         A_km=2.0 * (k - 1) / (k * (k - 2)), alpha_km=km_limit(k)))
    return rows, best_regular_graphs


def optimized_on_support(n, H, alpha, n_extra=0):
    """Optimize the conductances of a fixed support H (version-1 optimizer).

    Returns the best objective over the uniform start on H. This tightens the
    uniform-weight bound: a feasible state on H with R < R*_star at alpha
    certifies that the global optimum at alpha is not a tree.
    """
    edges = sorted((min(i, j), max(i, j)) for i, j in H.edges())
    prob = core.Problem(n, edges, alpha)
    m = len(edges)
    r = core.multistart(prob, [np.full(m, 1.0 / m)], ["uniform"], prune=False)
    return r


def certify(n, H, alpha):
    """Return (R_H_uniform, R_H_optimized, R_star) at alpha."""
    A = mean_resistance_unweighted(H, n)
    m = H.number_of_edges()
    r = optimized_on_support(n, H, alpha)
    return A * m ** (1.0 / alpha), r["best"], star_R(n, alpha)
