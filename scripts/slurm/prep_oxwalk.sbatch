#!/bin/bash
# CPU-only prep (run once): builds the data the OxWalk / combined models need.
#   1) motion-energy realigned OxWalk wrist labels -> data/oxwalk/OxWalk_Dec2022_peakvalley_025s/Wrist_25Hz
#   2) OxWalk wrist resampled 25 -> 15 Hz          -> data/oxwalk_wrist_15hz
# Idempotent: a step is skipped only if its output holds all the expected CSV files.
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
count() { ls "$1"/*.csv 2>/dev/null | wc -l; }   # 0 if missing; never fails
[ -d "$SRC/Wrist_25Hz" ] || { log "ERROR: $SRC/Wrist_25Hz not found (cwd=$PWD)"; exit 1; }
N=$(count "$SRC/Wrist_25Hz")
log "input: $N OxWalk wrist 25 Hz files"

log "STEP 1/2: motion-energy realigned labels (wrist only) -> $REAL/Wrist_25Hz"
if [ "$(count "$REAL/Wrist_25Hz")" -ge "$N" ]; then
  log "complete ($N files), skipping"
else
  if [ -e "$REAL" ]; then
    log "found incomplete $REAL ($(count "$REAL/Wrist_25Hz") wrist files); moving it aside"
    mv "$REAL" "$REAL.incomplete.$SLURM_JOB_ID"
  fi
  TMP="$REAL.tmp.$SLURM_JOB_ID"          # built here, renamed only on success
  python -u step_counting/oxwalk/preprocessing/relabel_motion_energy_025s.py \
    --src "$SRC" --dst "$TMP" --radius_sec 0.25 --only Wrist_25Hz
  mv "$TMP" "$REAL"
fi

log "STEP 2/2: resample 25 -> 15 Hz -> $HZ15"
if [ "$(count "$HZ15")" -ge "$N" ]; then
  log "complete ($N files), skipping"
else
  python -u step_counting/cross_dataset/preprocessing/resample_oxwalk_25to15hz.py \
    --src "$SRC/Wrist_25Hz" --dst "$HZ15"
fi

R=$(count "$REAL/Wrist_25Hz"); H=$(count "$HZ15")
log "VERIFY: realigned wrist files: $R | 15 Hz files: $H (expected $N each)"
[ "$R" -ge "$N" ] && [ "$H" -ge "$N" ] || { log "ERROR: incomplete output"; exit 1; }
log "DONE"
