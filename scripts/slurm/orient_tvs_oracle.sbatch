#!/bin/bash
# Oracle axis-swap sweep on TVS: every recording under all 48 signed axis permutations, one array task per model (9 tasks, 3 at a time).
#   sbatch scripts/slurm/orient_tvs_oracle.sbatch                                  # real run
#   LIMIT=3 sbatch --array=0 --time=00:30:00 scripts/slurm/orient_tvs_oracle.sbatch   # smoke test (3 participants), then:
#   rm analysis/cache/orient_tvs_rotations_*.csv
# Override the GPU type if needed:  sbatch --gres=gpu:tesa100:1 --partition=gpu scripts/slurm/orient_tvs_oracle.sbatch
#SBATCH --job-name=tss_oracle
#SBATCH --array=0-8%3
#SBATCH --partition=bstgpu,bstgpu2,bstgpu3,gpu          # <-- only GPU workloads belong here
#SBATCH --gres=gpu:tesh100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
source ~/.bash_profile
set -euo pipefail
source "scripts/slurm/env.sh"
nvidia-smi -L || true
MODELS=(all3 clemson combined oxwalk orient_xyz orient_xyz_vm orient_xyz_rot orient_xyz_mirror orient_xyz_vm_mirror)
python analysis/orient_tvs_oracle.py --device cuda:0 --models ${MODELS[$SLURM_ARRAY_TASK_ID]} ${LIMIT:+--limit $LIMIT}
