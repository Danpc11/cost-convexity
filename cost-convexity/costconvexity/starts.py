"""Initial share vectors for the multistart search.

legacy_starts reproduces the version-1 start set exactly (same order, same
random draws for a given seed): uniform, minimum spanning tree, depth-first
tree, hub star, then n_random Dirichlet(0.35) allocations.

looped_starts is new. It supplies sparse looped supports built from the
mechanism identified in revision: just above the loop collapse, the best states
on the complete candidate graph are near-regular graphs of degree 3 to 5 with
nearly uniform conductance. Blind Dirichlet sampling in hundreds of
dimensions reaches these basins rarely, and at n = 12 it misses a cubic state
that beats the star. The looped starts use a separate random stream, so adding
them never changes the legacy draws.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

FLOOR = 1e-7


def _indicator(edges, subgraph_edges, floor=FLOOR):
    idx = {e: k for k, e in enumerate(edges)}
    q = np.full(len(edges), floor)
    for i, j in subgraph_edges:
        q[idx[(min(i, j), max(i, j))]] = 1.0
    return q / q.sum()


def legacy_starts(n, edges, rng, n_random):
    """Version-1 start set (shares, not logits), with labels."""
    m = len(edges)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    out = [np.full(m, 1.0 / m)]
    labels = ["uniform"]
    for name, tree in (("mst", nx.minimum_spanning_tree(G)),
                       ("dfs", nx.dfs_tree(G, 0).to_undirected())):
        out.append(_indicator(edges, tree.edges()))
        labels.append(name)
    hub = max(range(n), key=lambda v: G.degree(v))
    out.append(_indicator(edges, G.edges(hub)))
    labels.append("star")
    for k in range(n_random):
        out.append(rng.dirichlet(np.full(m, 0.35)))
        labels.append(f"dirichlet{k}")
    # Version 1 floors shares at 1e-15 before taking logs; multistart does the same.
    return out, labels


def mean_resistance_unweighted(G, n=None):
    """A(G): mean pairwise effective resistance with unit conductances."""
    n = G.number_of_nodes() if n is None else n
    L = nx.laplacian_matrix(G, nodelist=range(n)).toarray().astype(float)
    ev = np.linalg.eigvalsh(L)[1:]
    if ev[0] <= 1e-10:
        return np.inf
    return 2.0 * float(np.sum(1.0 / ev)) / (n - 1)


def swap_descent(H, n, rng, sweeps=None):
    """Lower A(H) by degree-preserving double-edge swaps (greedy, first improvement).

    Each proposal replaces edges (a, b), (c, d) by (a, d), (c, b); it is
    accepted when the graph stays simple and connected and A decreases. The
    degree sequence, hence the edge count, never changes.
    """
    H = nx.Graph(H)
    A = mean_resistance_unweighted(H, n)
    if sweeps is not None:
        tries = sweeps
    else:
        # each proposal costs one O(n^3) eigendecomposition, so the budget shrinks with n
        cap = 6000 if n <= 128 else 1500 if n <= 256 else 300 if n <= 512 else 60
        tries = min(20 * H.number_of_edges(), cap)
    edges = list(H.edges())
    for _ in range(tries):
        i, j = rng.choice(len(edges), 2, replace=False)
        (a, b), (c, d) = edges[i], edges[j]
        if rng.random() < 0.5:
            c, d = d, c
        if len({a, b, c, d}) < 4 or H.has_edge(a, d) or H.has_edge(c, b):
            continue
        H.remove_edges_from([(a, b), (c, d)])
        H.add_edges_from([(a, d), (c, b)])
        A_new = mean_resistance_unweighted(H, n) if nx.is_connected(H) else np.inf
        if A_new < A - 1e-13:
            A = A_new
            edges[i], edges[j] = (a, d), (c, b)
        else:
            H.remove_edges_from([(a, d), (c, b)])
            H.add_edges_from([(a, b), (c, d)])
    return A, H


def best_regular_graphs(n, ks=(3, 4, 5, 6, 7, 8), samples=40, seed=0, descent=True):
    """For each degree k, a k-regular graph on n nodes with low A(G).

    The best of `samples` random k-regular graphs is improved by
    degree-preserving swap descent. The result is a good, not a certified
    optimal, low-resistance regular graph; every such graph is an explicit
    feasible state and so still yields a certified bound (ansatz.py).
    """
    rng = np.random.default_rng(seed)
    best = {}
    for k in ks:
        if k >= n or (n * k) % 2:
            continue
        cand = None
        for _ in range(samples):
            H = nx.random_regular_graph(k, n, seed=int(rng.integers(2**31)))
            if not nx.is_connected(H):
                continue
            a = mean_resistance_unweighted(H, n)
            if cand is None or a < cand[0]:
                cand = (a, H)
        if cand is not None and descent:
            cand = swap_descent(cand[1], n, rng)
        if cand is not None:
            best[k] = cand
    return best


def sparse_subgraphs(n, edges, degrees=(3, 4, 5, 6), samples=40, seed=0):
    """Sparse connected spanning subgraphs of a general candidate graph.

    For each target mean degree k: a random spanning tree (minimum spanning
    tree under random weights) plus random candidate edges up to round(k n / 2)
    edges; the sample with the lowest A(G) is kept. On the complete graph
    best_regular_graphs is used instead.
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    m = len(edges)
    best = {}
    for k in degrees:
        target = int(round(k * n / 2))
        if target <= n - 1 or target >= m:
            continue
        cand = None
        for _ in range(samples):
            for u, v in G.edges():
                G[u][v]["rw"] = rng.random()
            T = nx.minimum_spanning_tree(G, weight="rw")
            H = nx.Graph(T)
            rest = [e for e in edges if not H.has_edge(*e)]
            pick = rng.choice(len(rest), size=target - (n - 1), replace=False)
            H.add_edges_from(rest[i] for i in pick)
            a = mean_resistance_unweighted(H, n)
            if cand is None or a < cand[0]:
                cand = (a, H)
        if cand is not None:
            best[k] = cand
    return best


REGULAR_DEGREES = (3, 4, 5, 6, 7, 8, 10, 12, 16)


def looped_starts(n, edges, ks=REGULAR_DEGREES, samples=40, seed=12345):
    """Mechanism-informed looped starts, with labels such as 'reg4' or 'sub4'."""
    complete = len(edges) == n * (n - 1) // 2
    if complete:
        graphs = best_regular_graphs(n, ks, samples, seed)
        tag = "reg"
    else:
        graphs = sparse_subgraphs(n, edges, tuple(k for k in ks if k <= 8), samples, seed)
        tag = "sub"
    out, labels = [], []
    for k, (_, H) in sorted(graphs.items()):
        out.append(_indicator(edges, H.edges()))
        labels.append(f"{tag}{k}")
    return out, labels


def tree_starts(n, edges, n_random_roots=3, seed=0):
    """Breadth-first spanning trees rooted at the graph center(s), at node 0
    (the sink of the Corson loads) and at a few random nodes.

    On the complete graph a breadth-first tree is the star, which is the exact
    optimal tree (ansatz.py). On a lattice the optimal tree is not known in
    closed form, and these starts, together with the upward pass of
    sweep.sweep, keep the tree side of the comparison as strong as the looped
    side.
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    roots = list(dict.fromkeys(list(nx.center(G)) + [0] +
                               [int(r) for r in rng.choice(n, min(n, n_random_roots), replace=False)]))
    out, labels = [], []
    for r in roots:
        T = nx.bfs_tree(G, r).to_undirected()
        out.append(_indicator(edges, T.edges()))
        labels.append(f"bfs{r}")
    return out, labels
