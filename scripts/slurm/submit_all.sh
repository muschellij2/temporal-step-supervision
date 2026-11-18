#!/bin/bash
# Submit everything from the repo root:  bash scripts/slurm/submit_all.sh
# Smoke test first:                      EPOCHS=1 bash scripts/slurm/submit_all.sh
source ~/.bash_profile
# Must be executed, not sourced: `set -e` below would otherwise end your login shell.
if [ "${BASH_SOURCE[0]}" != "$0" ]; then
  echo "Run this with:  bash scripts/slurm/submit_all.sh   (do not 'source' it)" >&2
  return 1
fi
set -euo pipefail
mkdir -p logs   # SLURM does not create the --output directory
prep=$(sbatch --parsable scripts/slurm/prep_oxwalk.sbatch)
c=$(sbatch --parsable scripts/slurm/fit_clemson.sbatch)
o=$(sbatch --parsable --dependency=afterok:$prep scripts/slurm/fit_oxwalk.sbatch)
b=$(sbatch --parsable --dependency=afterok:$prep scripts/slurm/fit_combined.sbatch)
echo "prep=$prep clemson=$c oxwalk=$o combined=$b"
