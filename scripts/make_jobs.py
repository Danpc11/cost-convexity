#!/usr/bin/env python
"""Write the revision job list for a single 45-core server.

Output: scripts/cluster/jobs.tsv with columns
    priority  workers  cost  tag  command
priority 1 runs first; within a priority, expensive jobs start first so that
short jobs can fill the remaining cores (scripts/cluster/run_pool.py).
`cost` is a rough relative estimate used only for that ordering.

Experiments
  E1   collapse exponent on K_n, n = 8 ... 1024 (isotropic load)
  E1s  seed robustness of E1 at n = 64, 128, 256
  E2   collapse on non-complete candidate graphs (lattices, random graphs; 3 seeds)
  E3   Corson loads (sigma = 0.5, 1, 2) and mixtures from sink fluctuations
       (mix:0, Corson's sigma -> inf) to isotropic load (mix:1 = E1/E2)
  E5   support growth at fixed alpha (Table IV, Fig. 4), n up to 256
  E6   full (alpha, n) scan on K_n for the phase-diagram panel
  E4   certified bounds (ansatz_bound.py) and the uniform-weight phase prediction
"""
import sys
from pathlib import Path

PY = "python scripts/run_sweep.py"
jobs = []


def add(pri, workers, cost, exp, graph, load="iso", amax=0.75, amin=0.30, step=0.005,
        extra="", tag=None):
    tag = tag or f"{graph}__{load}".replace(":", "-")
    cmd = (f"{PY} --graph {graph} --load {load} --amax {amax} --amin {amin} --step {step} "
           f"--out results/{exp} --tag {tag} --workers {workers} {extra}").strip()
    jobs.append((pri, workers, cost, f"{exp}/{tag}", cmd))


def kn_cost(n):
    return (n / 64.0) ** 3


# ---------------------------------------------------------------- E1
for n in (8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64):
    add(1, 2, kn_cost(n), "E1", f"complete:{n}", amax=0.72, amin=0.45)
for n in (80, 96, 128):
    add(1, 6, kn_cost(n), "E1", f"complete:{n}", amax=0.72, amin=0.55)
for n in (192, 256):
    add(1, 12, kn_cost(n), "E1", f"complete:{n}", amax=0.72, amin=0.60,
        extra="--n-random 12 --looped-samples 12")
for n in (384, 512):
    add(1, 16, kn_cost(n), "E1", f"complete:{n}", amax=0.72, amin=0.62,
        extra="--n-random 8 --looped-samples 6")
for n in (768, 1024):          # optional tier: hours each, runs last
    add(6, 22, kn_cost(n), "E1", f"complete:{n}", amax=0.72, amin=0.64, step=0.005,
        extra="--n-random 4 --looped-samples 3")

# ---------------------------------------------------------------- E1s
for n in (64, 128, 256):
    for sd in (101, 2029):
        add(4, 6 if n < 256 else 12, kn_cost(n), "E1s", f"complete:{n}",
            amax=0.70, amin=0.60, extra=f"--seed {sd}",
            tag=f"complete-{n}__iso__seed{sd}")

# ---------------------------------------------------------------- E2
def topo_workers(n):
    return 2 if n <= 64 else 4 if n <= 144 else 6


for L in (4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24):
    add(2, topo_workers(L * L), (L * L / 64) ** 2.5, "E2", f"square:{L}", amax=0.70, amin=0.08,
        step=0.01)
for L in (3, 4, 5, 6, 7, 8, 10, 12, 14, 16):
    add(2, topo_workers(L * L // 2), (L * L / 128) ** 2.5, "E2", f"triangular:{L}", amax=0.75,
        amin=0.08, step=0.01)
for seed in (1, 2, 3):
    for n in (16, 24, 32, 48, 64, 96, 128, 192, 256):
        w, c = topo_workers(n), (n / 64) ** 2.5
        add(2, w, c, "E2", f"rrg:4:{n}:{seed}", amax=0.80, amin=0.08, step=0.01)
        if n > 16:
            add(2, w, c, "E2", f"rrg:8:{n}:{seed}", amax=0.85, amin=0.08, step=0.01)
        add(2, w, c, "E2", f"er:8:{n}:{seed}", amax=0.85, amin=0.08, step=0.01)
        add(2, w, c, "E2", f"ba:3:{n}:{seed}", amax=0.85, amin=0.08, step=0.01)
        add(2, w, c, "E2", f"rgg:{n}:{(10 / (3.1416 * n)) ** 0.5:.4f}:{seed}", amax=0.85,
            amin=0.08, step=0.01)

# ---------------------------------------------------------------- E3
for sigma in ("0.5", "1", "2"):          # sigma -> inf is mix:0 below
    for L in (4, 5, 6, 7, 8, 10, 12, 14, 16):
        add(3, topo_workers(L * L), (L * L / 64) ** 2.5, "E3", f"square:{L}",
            load=f"corson:{sigma}", amax=1.00, amin=0.08, step=0.01)
    for n in (8, 16, 24, 32, 48, 64, 96, 128):
        add(3, topo_workers(n), kn_cost(n), "E3", f"complete:{n}", load=f"corson:{sigma}",
            amax=1.00, amin=0.40, step=0.01)
for lam in ("0", "0.25", "0.5", "0.75", "0.9", "0.99"):
    for n in (8, 16, 32, 64, 128):
        add(3, topo_workers(n), kn_cost(n), "E3", f"complete:{n}", load=f"mix:{lam}",
            amax=1.00, amin=0.40, step=0.01)
    for L in (4, 6, 8, 12, 16):
        add(3, topo_workers(L * L), (L * L / 64) ** 2.5, "E3", f"square:{L}",
            load=f"mix:{lam}", amax=1.00, amin=0.08, step=0.01)

# ---------------------------------------------------------------- E5
for n in (8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 128, 192, 256):
    w = 2 if n <= 64 else 6 if n <= 128 else 12
    add(4, w, kn_cost(n), "E5", f"complete:{n}", amax=0.90, amin=0.55, step=0.05,
        extra="--confirm 99")

# ---------------------------------------------------------------- E6
for n in (8, 12, 16, 24, 32, 48, 64, 96, 128):
    w = 2 if n <= 32 else 4 if n <= 64 else 8
    add(5, w, kn_cost(n) * 3, "E6", f"complete:{n}", amax=0.98, amin=0.50, step=0.01,
        extra="--confirm 99")

# ---------------------------------------------------------------- E4 (single core each)
jobs.append((1, 1, 50.0, "E4/ansatz_bound",
             "python scripts/ansatz_bound.py --sizes 8,12,16,20,24,28,32,40,48,56,64,80,96,128,"
             "192,256,384,512,768,1024,1536,2048,3072,4096 --n-opt-max 256 "
             "--out results/E4/ansatz_bound.tsv"))
jobs.append((1, 1, 20.0, "E4/phase_prediction",
             "python scripts/phase_prediction.py --out results/E4/phase_prediction.tsv"))

out = Path(sys.argv[1] if len(sys.argv) > 1 else "scripts/cluster/jobs.tsv")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    f.write("priority\tworkers\tcost\ttag\tcommand\n")
    for j in jobs:
        f.write("\t".join(str(x) for x in j) + "\n")
print(f"{len(jobs)} jobs written to {out}; "
      f"total relative cost {sum(j[2] for j in jobs):.0f}")
