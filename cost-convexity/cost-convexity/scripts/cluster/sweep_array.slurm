#!/bin/bash
# SLURM array over the lines of scripts/cluster/jobs.txt.
#
#   cd cost-convexity
#   python scripts/make_jobs.py                     # writes scripts/cluster/jobs.txt
#   N=$(wc -l < scripts/cluster/jobs.txt)
#   sbatch --array=1-${N}%20 scripts/cluster/sweep_array.slurm
#
# Test first with a cheap slice, e.g. --array=1-3 (complete graph, n = 8, 12, 16).
# Adjust partition, time and the environment block to your cluster.
#SBATCH --job-name=costconv
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
##SBATCH --partition=general

set -euo pipefail
mkdir -p logs

# ---- environment (edit) ------------------------------------------------------
# module load anaconda3
# conda activate costconv
# -----------------------------------------------------------------------------

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" scripts/cluster/jobs.txt)
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} on $(hostname): ${CMD}"
eval "${CMD} --workers ${SLURM_CPUS_PER_TASK}"
echo "[$(date)] done"
