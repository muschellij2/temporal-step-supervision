#!/bin/bash
# Orientation-robustness experiments, one array task per arm (8 arms). Each task trains cv (5 folds) + loco (3) + full (1)
# = 9 models. Submit:  bash scripts/slurm/submit_gpu.sh scripts/slurm/orient.sbatch
# Just the final fits (no CV), augmented arms only (task ids: xyz_rot=1 xyz_vm_rot=4 xyz_mirror=5 xyz_vm_mirror=6 xyz_vm=7):
#   PROTOCOLS=full bash scripts/slurm/submit_gpu.sh scripts/slurm/orient.sbatch --array=0,5,6,7      # the 2x2: xyz, xyz_mirror, xyz_vm_mirror, xyz_vm
# Add the 24 MIRRORED mappings to finished CV/LOCO runs (non-destructive: writes *__mirror.csv, only scoring runs):
#   MIRROR_ONLY=1 PROTOCOLS="cv loco" sbatch --array=0,1,5,6,7 scripts/slurm/orient.sbatch
# Smoke test (1 epoch, 1 split, 1 rotation):  EPOCHS=1 ONLY=fold0 ROT=0 sbatch --array=0 scripts/slurm/orient.sbatch
#SBATCH --job-name=tss_orient
#SBATCH --array=0-7
#SBATCH --partition=bstgpu,bstgpu2,bstgpu3,gpu          # <-- only GPU workloads belong here
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
source ~/.bash_profile
set -euo pipefail
source "scripts/slurm/env.sh"
nvidia-smi -L || true

ARMS=(xyz xyz_rot vm xyz_vm_drop xyz_vm_rot xyz_mirror xyz_vm_mirror xyz_vm)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
python scripts/orient_cv.py --device cuda:0 --arms $ARM --protocol ${PROTOCOLS:-cv loco full} \
    --num_workers ${SLURM_CPUS_PER_TASK:-6} --epochs ${EPOCHS:-120} --rot_eval ${ROT:-all} \
    ${ONLY:+--only_split $ONLY} ${MIRROR_ONLY:+--mirror_only}
