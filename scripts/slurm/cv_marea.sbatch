#!/bin/bash
# 5-fold participant-level CV over Clemson+OxWalk+MAREA, two arms (with/without MAREA in training).
# One array task per fold (each does both arms). Needs prep_oxwalk.sbatch (data/oxwalk_wrist_15hz) and
# data/marea/ (python scripts/prepare_marea.py) first.  Smoke test: EPOCHS=1 sbatch --array=0 scripts/slurm/cv_marea.sbatch
#SBATCH --job-name=tss_cv_marea
#SBATCH --array=0-4
#SBATCH --partition=bstgpu,bstgpu2,bstgpu3,gpu          # <-- only GPU workloads belong here
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
source ~/.bash_profile
set -euo pipefail
source "scripts/slurm/env.sh"
nvidia-smi -L || true

python scripts/cv_with_marea.py --device cuda:0 --folds 5 --only_fold ${SLURM_ARRAY_TASK_ID} \
    --num_workers ${SLURM_CPUS_PER_TASK:-4} --epochs ${EPOCHS:-120}
