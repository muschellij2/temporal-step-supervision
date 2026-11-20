#!/bin/bash
# "Fuller" models: place8 and place16 augmentation trained on ALL available data = Clemson + OxWalk + MAREA + Mobilise-D TVS, plus 5-fold
# participant-level CV on the same four cohorts (out-of-sample results per cohort). One array task per arm (2 tasks): 0 xyz_place8, 1 xyz_place16.
#   sbatch scripts/slurm/orient_fuller.sbatch                                       # cv (5 models) + full (1 model) per arm
#   PROTOCOLS=full sbatch scripts/slurm/orient_fuller.sbatch                        # just the final fits
#   EPOCHS=1 ONLY=fold0 sbatch --array=0 --time=00:30:00 scripts/slurm/orient_fuller.sbatch   # smoke test; then: rm -rf runs/orient_fuller analysis/cache/orient_fuller
# Outputs: runs/orient_fuller/<protocol>/<arm>/<split>/model.pt ; analysis/cache/orient_fuller/cv_<arm>_fold<k>.csv (held-out scores, as recorded)
#SBATCH --job-name=tss_fuller
#SBATCH --array=0-1
#SBATCH --partition=bstgpu,bstgpu2,bstgpu3,gpu          # <-- only GPU workloads belong here
#SBATCH --gres=gpu:tesh100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
source ~/.bash_profile
set -euo pipefail
source "scripts/slurm/env.sh"
nvidia-smi -L || true

ARMS=(xyz_place8 xyz_place16)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
python scripts/orient_cv.py --device cuda:0 --arms $ARM --protocol ${PROTOCOLS:-cv full} \
    --cohorts clemson oxwalk marea tvs --tag orient_fuller --rot_eval 0 \
    --num_workers ${SLURM_CPUS_PER_TASK:-4} --epochs ${EPOCHS:-120} ${ONLY:+--only_split $ONLY}
