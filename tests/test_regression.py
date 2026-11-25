"""The rewritten core reproduces the version-1 module.

Run with: python -m pytest -q tests  (or python tests/test_regression.py)
"""
import sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "legacy"))
import os
(ROOT / "results").mkdir(exist_ok=True)
os.chdir(ROOT / "results")  # legacy module writes pre_results/ in the cwd
import pre_all as pa                       # noqa: E402
from costconvexity import core, starts, ansatz   # noqa: E402


def test_gradient_matches_legacy():
    rng = np.random.default_rng(1)
    for n in (9, 14):
        E = pa.edge_list(n)
        w = rng.uniform(0.2, 1.0, len(E))
        r0, g0 = pa.rbar_grad(n, pa.incidence(n, E), w)
        r1, g1 = core.Problem(n, E, 0.7).value_grad(w)
        assert abs(r1 - r0) / r0 < 1e-12
        assert np.max(np.abs(g1 - g0)) / np.max(np.abs(g0)) < 1e-10


def test_gradient_general_load_fd():
    from costconvexity import loads
    rng = np.random.default_rng(2)
    n = 10
    E = pa.edge_list(n)
    Q, sc = loads.build("corson:1.0", n)
    prob = core.Problem(n, E, 0.7, None, Q, sc)
    w = rng.uniform(0.2, 1.0, len(E))
    _, g = prob.value_grad(w)
    h = 1e-6
    num = np.array([(prob.value(w + h * np.eye(len(E))[k]) - prob.value(w - h * np.eye(len(E))[k])) / (2 * h)
                    for k in range(len(E))])
    assert np.max(np.abs(num - g)) / np.max(np.abs(num)) < 1e-6


def _compare(n, alpha, n_random, seed):
    E = pa.edge_list(n)
    t = time.time(); old = pa.multistart(n, E, alpha, n_random=n_random, seed=seed); t_old = time.time() - t
    rng = np.random.default_rng(seed)
    s, lab = starts.legacy_starts(n, E, rng, n_random)
    t = time.time(); new = core.multistart(core.Problem(n, E, alpha), s, lab); t_new = time.time() - t
    return old, new, t_old, t_new


def test_table_III_row():
    old, new, *_ = _compare(12, 0.65, 60, 11)
    assert abs(new["best"] - old["best"]) / old["best"] < 1e-9
    assert new["active"] == old["active"] == 18


def test_tree_at_32():
    old, new, t_old, t_new = _compare(32, 0.65, 24, 7)
    assert round(new["best"], 6) == 381.631590
    assert new["cycle_rank"] == 0
    print(f"n=32 timing legacy {t_old:.1f}s new {t_new:.1f}s")


def test_star_lemma():
    import networkx as nx
    for n in (8, 15):
        for a in (0.4, 0.7, 1.5):
            assert abs(ansatz.tree_optimal_R(nx.star_graph(n - 1), a) - ansatz.star_R(n, a)) < 1e-9 * ansatz.star_R(n, a)
            assert abs(ansatz.star_R(n, a) - pa.uniform_star_rbar(n, a)) < 1e-9 * ansatz.star_R(n, a)
            rng = np.random.default_rng(n)
            for _ in range(20):
                T = nx.random_labeled_tree(n, seed=int(rng.integers(1e9))) if hasattr(nx, "random_labeled_tree") else nx.random_tree(n, seed=int(rng.integers(1e9)))
                assert ansatz.tree_optimal_R(T, a) >= ansatz.star_R(n, a) * (1 - 1e-12)


def test_tree_formula_matches_optimizer():
    import networkx as nx
    n = 10
    T = nx.random_labeled_tree(n, seed=3) if hasattr(nx, "random_labeled_tree") else nx.random_tree(n, seed=3)
    E = sorted((min(i, j), max(i, j)) for i, j in T.edges())
    r = core.multistart(core.Problem(n, E, 0.6), [np.full(len(E), 1 / len(E))], prune=False)
    assert abs(r["best"] - ansatz.tree_optimal_R(T, 0.6)) / r["best"] < 1e-7


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            t = time.time(); f(); print(f"{name}: ok ({time.time()-t:.1f}s)")
