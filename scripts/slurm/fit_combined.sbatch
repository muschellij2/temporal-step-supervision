#!/bin/bash
# Combined Clemson+OxWalk model (NOT from the paper). Needs prep_oxwalk.sbatch first.
# Quick smoke test (1 epoch):  EPOCHS=1 sbatch scripts/slurm/fit_combined.sbatch
#SBATCH --job-name=tss_combined
#SBATCH --partition=bstgpu,bstgpu2,bstgpu3,gpu          # <-- only GPU workloads belong here
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
source ~/.bash_profile
set -euo pipefail
source "scripts/slurm/env.sh"
nvidia-smi -L || true

python scripts/fit_final_combined.py --device cuda:0 --num_workers ${SLURM_CPUS_PER_TASK:-4} --epochs ${EPOCHS:-120}
