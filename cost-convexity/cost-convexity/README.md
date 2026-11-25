# cost-convexity

Code for **"Cost convexity controls the architecture of resistance-optimal networks"**
(H. Tovar, G. Vázquez-Victorio, E. Hernández-Lemus, A. Cervera, D. Pérez-Calixto;
submitted to *Physical Review E*, manuscript EW12546; preprint Research Square rs-11174611/v1).

We optimize the conductances `w_e` of a network to minimize a transport objective
`c tr(L(w)^+ Q)` under the resource law `sum_e a_e w_e^alpha = C0`. With isotropic
loading `Q = I - 11^T/n` the objective is the mean pairwise effective resistance.

## Contents

| path | what it is |
|---|---|
| `costconvexity/core.py` | objective, O(n^3) gradient for any load `Q`, multistart optimizer (same algorithm as version 1) |
| `costconvexity/starts.py` | version-1 start set; looped starts (low-resistance regular graphs, sparse subgraphs); breadth-first tree starts |
| `costconvexity/sweep.py` | loop-collapse exponent `alpha*`: downward sweep with continuation, then alternating continuation passes |
| `costconvexity/ansatz.py` | optimal-tree lemma (the star on `K_n`), uniform-weight competitors, certified upper bounds on `alpha*` |
| `costconvexity/graphs.py`, `loads.py` | candidate graphs and load matrices from short text specifications |
| `scripts/run_sweep.py` | command-line front end: one sweep per call |
| `scripts/make_jobs.py` | writes the revision job list (`scripts/cluster/jobs.txt`) |
| `scripts/ansatz_bound.py` | certified bounds on `alpha*(n)` up to n = 4096 |
| `scripts/collect.py` | gathers summaries, draws the revision figures |
| `scripts/cluster/` | SLURM array script, bound job, and a scheduler-free runner |
| `legacy/` | the version-1 supplement (`pre_all.py` and the executed notebook), unchanged |
| `tests/` | regression tests: the new core reproduces version-1 states |

## Install

```bash
conda env create -f environment.yml && conda activate costconv
# or: pip install -r requirements.txt
python tests/test_regression.py
```

## Revision experiments

| id | question | specification |
|---|---|---|
| E1 | `alpha*(n)` on the complete candidate graph with mechanism-informed starts, n = 8 to 256 | `complete:N`, load `iso` |
| E2 | does the size dependence persist on non-complete candidate graphs? | `square:L`, `triangular:L`, `rrg:D:N`, `er:C:N`, `ba:M:N`, `rgg:N:R` |
| E3 | loading distribution, candidate topology, or their interaction? | `corson:SIGMA` (Corson 2010), `mix:LAMBDA` (sink-star to isotropic) on `complete` and `square` |
| E4 | certified upper bound on `alpha*(n)`, n up to 4096 | `scripts/ansatz_bound.py` |

```bash
python scripts/make_jobs.py
N=$(wc -l < scripts/cluster/jobs.txt)
sbatch --array=1-${N}%20 scripts/cluster/sweep_array.slurm
sbatch scripts/cluster/ansatz.slurm
# when done
python scripts/collect.py
```

A single sweep, for example:

```bash
python scripts/run_sweep.py --graph complete:64 --load iso --amax 0.72 --amin 0.45 --workers 32 --out results/E1
```

Each sweep writes `<tag>.tsv` (every exponent visited: objective, active edges,
cycle rank, degrees, which start won, gap to the blind version-1 protocol),
`<tag>_shares.npz` (the budget shares of every state) and `<tag>_summary.json`.

## Scope of the numerical claims

Below `alpha = 1` the problem is nonconvex. Every looped or tree state reported
is the best state found with a stated search, not a certified global optimum.
Two statements are certified:

* the star is the unique optimal spanning tree of `K_n` at every `alpha` (closed form, `ansatz.py`);
* any explicit looped state that beats the star at `alpha` proves the global optimum there is not
  a tree, so the regular-graph competitors give certified upper bounds on `alpha*(n)`.

## License and citation

MIT (see `LICENSE`). Please cite the article and the archived release (`CITATION.cff`).
