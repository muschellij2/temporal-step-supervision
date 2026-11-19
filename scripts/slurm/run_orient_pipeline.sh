#!/bin/bash
# Full (non-smoke) orientation / VM / augmentation pipeline on JHPCE. Run from the repo root AFTER `git pull`, with
# data/mobilised_tvs, data/marea, data/clemson, data/oxwalk_wrist_15hz already synced.
#
#   bash scripts/slurm/run_orient_pipeline.sh            # submits everything (prints each job id)
#   DRY=1 bash scripts/slurm/run_orient_pipeline.sh      # only prints the sbatch commands
#   GRES=gpu:tesa100:1 PART=gpu bash scripts/slurm/run_orient_pipeline.sh     # use an A100 instead of the default H100
#
# Stages (each is resumable: finished models / scored CSVs are skipped, so re-running just continues):
#   1  FIT   final models on all 3 cohorts for the 2x2 {VM channel} x {rotation+mirror augmentation} + xyz_rot:
#              task ids (scripts/slurm/orient.sbatch): 0 xyz | 1 xyz_rot | 5 xyz_mirror | 6 xyz_vm_mirror | 7 xyz_vm
#   2  TVS   score ALL models (clemson, combined, all3, oxwalk + the orient arms of stage 1) on the Mobilise-D TVS wrist
#            recordings as recorded (no rotation chosen); starts when stage 1 finishes OK
#   3  CV    5-fold participant CV of the same arms, each held-out recording also scored under all 24 rotations
#   4  LOCO  leave-one-cohort-out of the same arms (orientation shift test; held-out MAREA has no running in training)
# Only 2 array tasks run at once in stages 3-4 (%2) so the queue is not flooded. One model ~ one GPU for <= 24 h; if your
# first stage-1 log shows a model takes more than ~2.5 h, stages 3/4 (5 resp. 3 models per task) need a longer --time.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

GRES="${GRES:-gpu:tesh100:1}"                  # smoke-tested type; l40s/tesa100 also work (CC >= 8.0)
PART="${PART:-bstgpu2,gpu}"                    # H100 lives in bstgpu2 and gpu; for tesa100 use PART=gpu; l40s: PART=bstgpu3,gpu
CPUS="${CPUS:-4}"
MEM="${MEM:-32G}"
ARMS_ARRAY="${ARMS_ARRAY:-0,1,5,6,7}"          # xyz xyz_rot xyz_mirror xyz_vm_mirror xyz_vm
mkdir -p logs

run() {   # run <env assignments...> -- <sbatch args...>   -> prints the job id
    if [ -n "${DRY:-}" ]; then echo "[dry] $*" >&2; echo "DRYJOB"; else "$@"; fi
}
SB=(sbatch --parsable --gres="$GRES" --partition="$PART" --cpus-per-task="$CPUS" --mem="$MEM")

echo "== stage 1: final fits (arms $ARMS_ARRAY)"
J1=$(run env PROTOCOLS=full "${SB[@]}" --array="$ARMS_ARRAY" scripts/slurm/orient.sbatch)
echo "   job $J1"

echo "== stage 2: TVS scoring (after stage 1 succeeds)"
J2=$(run sbatch --parsable --dependency=afterok:"$J1" --partition="$PART" --gres="$GRES" --cpus-per-task=2 --mem=16G \
    --time=03:00:00 --job-name=tss_orient_tvs --output=logs/orient_tvs_%j.out \
    --wrap='source ~/.bash_profile; source scripts/slurm/env.sh; python analysis/orient_tvs.py --device cuda:0')
echo "   job $J2"

echo "== stage 3: 5-fold CV"
J3=$(run env PROTOCOLS=cv "${SB[@]}" --array="${ARMS_ARRAY}%2" scripts/slurm/orient.sbatch)
echo "   job $J3"

echo "== stage 4: leave-one-cohort-out"
J4=$(run env PROTOCOLS=loco "${SB[@]}" --array="${ARMS_ARRAY}%2" scripts/slurm/orient.sbatch)
echo "   job $J4"

cat <<EOF

Submitted: fit=$J1  tvs=$J2  cv=$J3  loco=$J4      (watch: squeue -u \$USER)
When they finish, copy these back to your laptop (results are gitignored, so rsync them), then run targets there:
  rsync -av \$jtenig:/dcs05/ciprian/smart/temporal-step-supervision/analysis/cache/orient_tvs_participants.csv analysis/cache/
  rsync -av \$jtenig:/dcs05/ciprian/smart/temporal-step-supervision/analysis/cache/orient_tvs_recordings.csv   analysis/cache/
  rsync -av \$jtenig:/dcs05/ciprian/smart/temporal-step-supervision/analysis/cache/orient/                      analysis/cache/orient/
  Rscript -e 'targets::tar_make()'        # rebuilds paper/paper.pdf with the new numbers filled in
EOF
