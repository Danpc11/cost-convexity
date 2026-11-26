"""Candidate graphs addressed by a short text specification.

    complete:N              K_N
    square:L                L x L square lattice (N = L^2); node 0 is a corner
    triangular:L            triangular lattice, networkx triangular_lattice_graph(L, L)
    rrg:D:N[:SEED]          random D-regular graph
    er:C:N[:SEED]           Erdos-Renyi G(N, C/(N-1)), resampled until connected
    rgg:N:RADIUS[:SEED]     random geometric graph in the unit square, resampled
                            until connected; edge cost a_e = 1 (use rgg_len for a_e = l_e)
    rgg_len:N:RADIUS[:SEED] the same graph with a_e = Euclidean length
    ba:M:N[:SEED]           Barabasi-Albert graph with M edges per new node

Every builder returns (n, edges, edge_cost or None, positions or None).
"""
from __future__ import annotations

import networkx as nx
import numpy as np


def _relabel(G):
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    return G


def _edges(G):
    return sorted((min(i, j), max(i, j)) for i, j in G.edges())


def build(spec):
    parts = spec.split(":")
    kind = parts[0]
    pos = None
    cost = None
    if kind == "complete":
        n = int(parts[1])
        G = nx.complete_graph(n)
    elif kind == "square":
        L = int(parts[1])
        H = nx.grid_2d_graph(L, L)
        pos0 = {v: v for v in H.nodes()}
        G = nx.convert_node_labels_to_integers(H, ordering="sorted", label_attribute="xy")
        pos = {i: G.nodes[i]["xy"] for i in G.nodes()}
        del pos0
    elif kind == "triangular":
        L = int(parts[1])
        H = nx.triangular_lattice_graph(L, L)
        G = nx.convert_node_labels_to_integers(H, ordering="sorted", label_attribute="xy")
        pos = {i: H.nodes[G.nodes[i]["xy"]]["pos"] for i in G.nodes()}
    elif kind == "rrg":
        d, n = int(parts[1]), int(parts[2])
        seed = int(parts[3]) if len(parts) > 3 else 1
        while True:
            G = nx.random_regular_graph(d, n, seed=seed)
            if nx.is_connected(G):
                break
            seed += 1000
    elif kind == "er":
        c, n = float(parts[1]), int(parts[2])
        seed = int(parts[3]) if len(parts) > 3 else 1
        while True:
            G = nx.gnp_random_graph(n, c / (n - 1), seed=seed)
            if nx.is_connected(G):
                break
            seed += 1000
    elif kind in ("rgg", "rgg_len"):
        n, radius = int(parts[1]), float(parts[2])
        seed = int(parts[3]) if len(parts) > 3 else 1
        while True:
            G = nx.random_geometric_graph(n, radius, seed=seed)
            if nx.is_connected(G):
                break
            seed += 1000
        pos = nx.get_node_attributes(G, "pos")
    elif kind == "ba":
        mm, n = int(parts[1]), int(parts[2])
        seed = int(parts[3]) if len(parts) > 3 else 1
        G = nx.barabasi_albert_graph(n, mm, seed=seed)
    else:
        raise ValueError(f"unknown graph specification {spec!r}")
    edges = _edges(G)
    n = G.number_of_nodes()
    if kind == "rgg_len":
        cost = np.array([np.hypot(*(np.subtract(pos[i], pos[j]))) for i, j in edges])
    return n, edges, cost, pos
