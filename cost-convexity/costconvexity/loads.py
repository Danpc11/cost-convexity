"""Load matrices Q for the objective c tr(L^+ Q).

    iso                 all node pairs weighted equally (Q = I - 11^T/n); the
                        objective of the manuscript
    pair:S:T            one source-sink pair
    corson:SIGMA[:SINK] Corson's fluctuating sources (PRL 104, 048703, 2010):
                        every node k != SINK injects i_k with <i_k> = 1 and
                        <i_k i_l> = 1 + SIGMA^2 delta_kl; the sink absorbs the
                        sum. SIGMA may be 'inf', which keeps only the
                        fluctuating part (sink-star loading).
    mix:LAMBDA[:SINK]   Q = (1 - LAMBDA) Q_sink + LAMBDA Q_iso, each normalized
                        to unit trace: an interpolation between loading
                        concentrated on one node (LAMBDA = 0) and isotropic
                        loading (LAMBDA = 1)

Every load is returned with the normalization used by Problem: None for
isotropic loading, otherwise (Q, scale) with scale = 1. The overall constant
does not affect the optimal state.
"""
from __future__ import annotations

import numpy as np


def corson_Q(n, sigma, sink=0):
    src = [k for k in range(n) if k != sink]
    A = np.zeros((n, n - 1))
    for c, k in enumerate(src):
        A[k, c] = 1.0
        A[sink, c] = -1.0
    if np.isinf(sigma):
        C = np.eye(n - 1)
    else:
        C = np.ones((n - 1, n - 1)) + sigma ** 2 * np.eye(n - 1)
    return A @ C @ A.T


def iso_Q(n):
    return np.eye(n) - np.ones((n, n)) / n


def build(spec, n):
    parts = spec.split(":")
    kind = parts[0]
    if kind == "iso":
        return None, None
    if kind == "pair":
        s, t = int(parts[1]), int(parts[2])
        b = np.zeros(n)
        b[s], b[t] = 1.0, -1.0
        return np.outer(b, b), 1.0
    if kind == "corson":
        sigma = float(parts[1])
        sink = int(parts[2]) if len(parts) > 2 else 0
        Q = corson_Q(n, sigma, sink)
        return Q / np.trace(Q), 1.0
    if kind == "mix":
        lam = float(parts[1])
        sink = int(parts[2]) if len(parts) > 2 else 0
        Qs = corson_Q(n, np.inf, sink)
        Qs /= np.trace(Qs)
        Qi = iso_Q(n)
        Qi /= np.trace(Qi)
        return (1 - lam) * Qs + lam * Qi, 1.0
    raise ValueError(f"unknown load specification {spec!r}")
