#!/bin/bash
# Certified upper bounds on alpha*(n) from k-regular competitors (E4), n up to 4096.
#SBATCH --job-name=costconv-ansatz
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
mkdir -p logs
# module load anaconda3; conda activate costconv
export OMP_NUM_THREADS=1
python scripts/ansatz_bound.py \
  --sizes 8,12,16,20,24,28,32,40,48,56,64,80,96,128,192,256,384,512,768,1024,1536,2048,3072,4096 \
  --n-opt-max 256 --out results/E4/ansatz_bound.tsv
