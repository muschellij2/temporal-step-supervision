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
export PYTHONUNBUFFERED=1   # write progress to the log as it happens, not at exit
log() { echo "[$(date '+%F %T')] $*"; }

log "job $SLURM_JOB_ID on $(hostname) starting"
source "${SLURM_SUBMIT_DIR}/scripts/slurm/env.sh"

REAL=data/oxwalk/OxWalk_Dec2022_peakvalley_025s
HZ15=data/oxwalk_wrist_15hz
SRC=data/oxwalk/OxWalk_Dec2022
[ -d "$SRC/Wrist_25Hz" ] || { log "ERROR: $SRC/Wrist_25Hz not found (cwd=$PWD)"; exit 1; }
log "input: $(ls "$SRC/Wrist_25Hz"/*.csv | wc -l) OxWalk wrist 25 Hz files"

log "STEP 1/2: motion-energy realigned labels -> $REAL"
if [ ! -d "$REAL" ]; then
  python -u step_counting/oxwalk/preprocessing/relabel_motion_energy_025s.py \
    --src "$SRC" --dst "$REAL" --radius_sec 0.25
else
  log "already exists, skipping"
fi

log "STEP 2/2: resample 25 -> 15 Hz -> $HZ15"
if [ ! -d "$HZ15" ]; then
  python -u step_counting/cross_dataset/preprocessing/resample_oxwalk_25to15hz.py \
    --src "$SRC/Wrist_25Hz" --dst "$HZ15"
else
  log "already exists, skipping"
fi

log "VERIFY: realigned wrist files: $(ls "$REAL/Wrist_25Hz"/*.csv | wc -l) | 15 Hz files: $(ls "$HZ15"/*.csv | wc -l) (expected 39 each)"
log "DONE"
