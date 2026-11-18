#!/bin/bash
# Submit everything from the repo root:  bash scripts/slurm/submit_all.sh
# Smoke test first:                      EPOCHS=1 bash scripts/slurm/submit_all.sh
set -euo pipefail
mkdir -p logs   # SLURM does not create the --output directory
prep=$(sbatch --parsable scripts/slurm/prep_oxwalk.sbatch)
c=$(sbatch --parsable scripts/slurm/fit_clemson.sbatch)
o=$(sbatch --parsable --dependency=afterok:$prep scripts/slurm/fit_oxwalk.sbatch)
b=$(sbatch --parsable --dependency=afterok:$prep scripts/slurm/fit_combined.sbatch)
echo "prep=$prep clemson=$c oxwalk=$o combined=$b"
