#!/usr/bin/env python
"""Write the job list of the revision experiments, one command per line.

  E1  complete candidate graph, isotropic load, n = 8 ... 128 (+ 192, 256)
  E2  non-complete candidate graphs, isotropic load
  E5  support growth with n at fixed exponents (Table IV), revised starts
  E3  load ensembles (Corson fluctuations, sink-star, isotropic mixtures)
      on the complete graph and on the square lattice

Each line is independent; scripts/cluster/sweep_array.slurm runs line
$SLURM_ARRAY_TASK_ID of jobs.txt. Lines are ordered roughly by cost so that a
small array index range can be tested first.
"""
import sys
from pathlib import Path

PY = "python scripts/run_sweep.py"
jobs = []


def add(exp, graph, load="iso", amax=0.75, amin=0.30, step=0.005, extra=""):
    tag = f"{graph}__{load}".replace(":", "-")
    jobs.append(f"{PY} --graph {graph} --load {load} --amax {amax} --amin {amin} "
                f"--step {step} --out results/{exp} --tag {tag} {extra}".strip())


# E1: complete graph, isotropic. alpha* lies in [0.55, 0.70] for n <= 64; the
# certified bound keeps it below 0.70 at every n tested.
for n in (8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 128):
    add("E1", f"complete:{n}", amax=0.72, amin=0.45)
for n in (192, 256, 384, 512):
    add("E1", f"complete:{n}", amax=0.72, amin=0.58, extra="--n-random 8 --looped-samples 10")

# E2: non-complete candidate graphs, isotropic load. alpha* is much lower on
# lattices, so the grid extends down to 0.10 at a coarser step.
for L in (4, 5, 6, 7, 8, 10, 12, 14, 16):
    add("E2", f"square:{L}", amax=0.95, amin=0.10, step=0.01)
for L in (3, 4, 5, 6, 7, 8, 10, 12):
    add("E2", f"triangular:{L}", amax=0.95, amin=0.10, step=0.01)
for d in (4, 8, 16):
    for n in (16, 24, 32, 48, 64, 96, 128):
        if d < n:
            add("E2", f"rrg:{d}:{n}:1", amax=0.95, amin=0.10, step=0.01)
for n in (16, 24, 32, 48, 64, 96, 128):
    add("E2", f"er:8:{n}:1", amax=0.95, amin=0.10, step=0.01)
    add("E2", f"ba:3:{n}:1", amax=0.95, amin=0.10, step=0.01)
    # mean degree near 10 in the unit square: radius^2 * pi * n ~ 10
    add("E2", f"rgg:{n}:{(10 / (3.1416 * n)) ** 0.5:.4f}:1", amax=0.95, amin=0.10, step=0.01)

# E3: loads. Corson's square lattice with a corner sink, and the same loads on K_n.
for sigma in ("0.5", "1", "2", "inf"):
    for L in (4, 6, 8, 10, 12):
        add("E3", f"square:{L}", load=f"corson:{sigma}", amax=1.00, amin=0.10, step=0.01)
    for n in (8, 16, 24, 32, 48, 64):
        add("E3", f"complete:{n}", load=f"corson:{sigma}", amax=1.00, amin=0.30, step=0.005)
for lam in ("0.25", "0.5", "0.75", "0.9", "0.95", "0.99"):
    for n in (8, 16, 32, 64):
        add("E3", f"complete:{n}", load=f"mix:{lam}", amax=1.00, amin=0.30, step=0.005)
    for L in (4, 6, 8, 10):
        add("E3", f"square:{L}", load=f"mix:{lam}", amax=1.00, amin=0.10, step=0.01)

# E5: support growth (Table IV and Fig. 4) with the revised starts: a coarse
# downward ladder 0.90, 0.85, ..., 0.55 at each size, never stopping early.
for n in (8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 128):
    add("E5", f"complete:{n}", amax=0.90, amin=0.55, step=0.05, extra="--confirm 99")

out = Path(sys.argv[1] if len(sys.argv) > 1 else "scripts/cluster/jobs.txt")
out.write_text("\n".join(jobs) + "\n")
print(f"{len(jobs)} jobs written to {out}")
