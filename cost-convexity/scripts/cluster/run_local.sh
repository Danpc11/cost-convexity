#!/bin/bash
# Run jobs.txt without a scheduler, P jobs at a time with W workers each
# (P x W should not exceed the number of cores). Example: P=4 W=8 bash scripts/cluster/run_local.sh
set -euo pipefail
P=${P:-4}; W=${W:-8}
mkdir -p logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nl -nln scripts/cluster/jobs.txt | while read -r i cmd; do
  echo "${cmd} --workers ${W} > logs/job_${i}.log 2>&1"
done | xargs -P "${P}" -I{} bash -c "{}"
