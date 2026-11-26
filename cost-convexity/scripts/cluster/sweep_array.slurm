#!/bin/bash
# SLURM array over the lines of scripts/cluster/jobs.txt.
#
#   cd cost-convexity
#   python scripts/make_jobs.py                     # writes scripts/cluster/jobs.tsv
#   N=$(( $(wc -l < scripts/cluster/jobs.tsv) - 1 ))
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
CMD=$(awk -F'\t' -v i=$((SLURM_ARRAY_TASK_ID+1)) 'NR==i{print $5}' scripts/cluster/jobs.tsv)
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} on $(hostname): ${CMD}"
eval "${CMD}"
echo "[$(date)] done"
