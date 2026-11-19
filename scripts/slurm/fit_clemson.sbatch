#!/bin/bash
# Paper model - Clemson wrist, all 30 participants.
# Quick smoke test (1 epoch):  EPOCHS=1 sbatch scripts/slurm/fit_clemson.sbatch
#SBATCH --job-name=tss_clemson
#SBATCH --partition=bstgpu,bstgpu2,bstgpu3,gpu          # <-- only GPU workloads belong here
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
source "scripts/slurm/env.sh"
nvidia-smi -L || true

python scripts/fit_final_paper.py --dataset clemson --device cuda:0 --num_workers ${SLURM_CPUS_PER_TASK:-4} --epochs ${EPOCHS:-120}
