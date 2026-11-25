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
| `scripts/collect.py` | gathers sweep summaries into `results/summary_alpha_star.tsv` (quick look) |
| `scripts/revision_tables.py` | Tables III, V and VI with the revised search (and the Fig. 2 curve) |
| `scripts/controls.py` | support-threshold, start-budget, seed and finite-jump insertion controls |
| `scripts/figures_revision.py` | all figures of the revised manuscript, in the version-1 house style |
| `scripts/sm_tables.py` | Supplemental Tables S1-S6 as LaTeX |
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
| E1 | `alpha*(n)` on the complete candidate graph, n = 8 to 512 (optional 768, 1024) | `complete:N`, load `iso` |
| E1s | seed robustness of E1 at n = 64, 128, 256 | `--seed 101`, `--seed 2029` |
| E2 | collapse on non-complete candidate graphs, 3 seeds for random families | `square:L`, `triangular:L`, `rrg:D:N:S`, `er:C:N:S`, `ba:M:N:S`, `rgg:N:R:S` |
| E3 | load or topology? Corson loads and sink-to-isotropic mixtures | `corson:SIGMA`, `mix:LAMBDA` on `complete` and `square` |
| E4 | certified bounds to n = 4096 and the uniform-weight phase prediction | `ansatz_bound.py`, `phase_prediction.py` |
| E5 | support growth at fixed alpha (Table IV, Fig. 4), n to 256 | `complete:N`, ladder 0.90 ... 0.55 |
| E6 | full (alpha, n) scan for the phase diagram, Fig. 2(d) | `complete:N`, alpha 0.98 ... 0.50 |

### On one machine (no scheduler), e.g. 45 cores

```bash
python scripts/make_jobs.py                       # writes scripts/cluster/jobs.tsv (317 jobs)
python scripts/cluster/run_pool.py --cores 45 --dry-run | head     # check the queue
mkdir -p logs
nohup python scripts/cluster/run_pool.py --cores 45 --max-priority 5 > logs/pool.log 2>&1 &
# later, the optional K_768 and K_1024 tier:
nohup python scripts/cluster/run_pool.py --cores 45 > logs/pool2.log 2>&1 &
tail -f logs/pool.log          # progress; logs/pool_status.tsv lists running and queued jobs
```

The pool never runs more than `--cores` worker processes, starts the most
expensive jobs first and fills the remaining cores with short ones. Finished jobs
are skipped on restart, so it can be stopped (`pkill -f run_pool.py`; also
`pkill -f run_sweep.py`) and relaunched. Memory use is small (well under 1 GB per
worker even at n = 1024). With a SLURM cluster, `scripts/cluster/sweep_array.slurm`
still works on the commands in the last column of `jobs.tsv`.

When the runs finish (or at any point, with whatever has finished):

```bash
bash scripts/cluster/postprocess.sh
```

`postprocess.sh` regenerates Tables III, V, VI and the controls, then the figures
into `figures/` and the supplemental tables into `supplement/sm_tables.tex`.

`results_local_2026-11-25/` holds the smaller runs behind the first revised draft
(complete graph to n = 64, lattices and random graphs to n = 64, loads to n = 32,
bounds to n = 2048), for comparison.

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
