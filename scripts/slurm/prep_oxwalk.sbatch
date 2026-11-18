#!/bin/bash
# CPU-only prep (run once): builds the data the OxWalk / combined models need.
#   1) motion-energy realigned OxWalk labels  -> data/oxwalk/OxWalk_Dec2022_peakvalley_025s
#   2) OxWalk wrist resampled 25 -> 15 Hz      -> data/oxwalk_wrist_15hz
#SBATCH --job-name=tss_prep
#SBATCH --partition=shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/scripts/slurm/env.sh"

if [ ! -d data/oxwalk/OxWalk_Dec2022_peakvalley_025s ]; then
  python step_counting/oxwalk/preprocessing/relabel_motion_energy_025s.py \
    --src data/oxwalk/OxWalk_Dec2022 --dst data/oxwalk/OxWalk_Dec2022_peakvalley_025s --radius_sec 0.25
else echo "realigned labels already exist, skipping"; fi

if [ ! -d data/oxwalk_wrist_15hz ]; then
  python step_counting/cross_dataset/preprocessing/resample_oxwalk_25to15hz.py \
    --src data/oxwalk/OxWalk_Dec2022/Wrist_25Hz --dst data/oxwalk_wrist_15hz
else echo "15 Hz OxWalk already exists, skipping"; fi
