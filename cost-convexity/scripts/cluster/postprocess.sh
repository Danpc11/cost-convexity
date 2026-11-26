#!/bin/bash
# Tables, controls, figures and supplemental tables from whatever is in results/.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/revision_tables.py --out results/tables
python scripts/controls.py --out results/controls
python scripts/collect.py
python scripts/figures_revision.py --res results --out figures
python scripts/sm_tables.py --res results --out supplement/sm_tables.tex
