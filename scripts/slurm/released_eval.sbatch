#!/bin/bash
# Compare the original authors' released per-fold CV models (runs/temporal-step-supervision-main-weights.zip) with our
# final models on the external sets. Inference only, no training. One array task per job (4 tasks, all at once):
#   0 parity   tests/check_released_parity.py: released folds re-score their own held-out Clemson/OxWalk subjects
#   1 marea    analysis/marea_eval.py --only marea                 -> analysis/cache/marea_counts_released.csv (+ probs)
#   2 tvs      analysis/orient_tvs.py                              -> analysis/cache/orient_tvs_{recordings,participants}_released.csv
#   3 mobig    analysis/mobig_eval.py  (Mobi-G, ankle, EXTERNAL VALIDATION ONLY) -> analysis/cache/mobig_{counts,participants}.csv
# Models: released_clemson / released_oxwalk (fold ensembles; every fold also gets its own rows) + our clemson combined all3 oxwalk.
#
# Before submitting (login node; compute nodes may have no internet):
#   scp runs/temporal-step-supervision-main-weights.zip  jhpce:<repo>/runs/       # runs/ and *.pt are gitignored
#   python scripts/prepare_released_weights.py                                   # -> runs/released/{clemson,oxwalk}/fold_XX
#   python -I scripts/prepare_mobig.py                                           # downloads Mobi-G -> data/mobig/{recordings,manifest.csv}
# Submit (from the repo root):
#   bash scripts/slurm/submit_gpu.sh scripts/slurm/released_eval.sbatch
#   LIMIT=2 bash scripts/slurm/submit_gpu.sh scripts/slurm/released_eval.sbatch --array=1-3 --time=00:30:00   # smoke test
#   bash scripts/slurm/submit_gpu.sh scripts/slurm/released_eval.sbatch --array=3                            # just Mobi-G
# Smoke-test outputs carry the _smoke tag and are safe to delete:  rm analysis/cache/*_smoke*
# PARITY_STRICT=1 makes task 0 fail the job on a parity mismatch (default: report only; see its log).
#SBATCH --job-name=tss_released
#SBATCH --array=0-3
#SBATCH --partition=bstgpu,bstgpu2,bstgpu3,gpu          # <-- only GPU workloads belong here
#SBATCH --gres=gpu:tesh100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
source ~/.bash_profile
set -euo pipefail
source "scripts/slurm/env.sh"
nvidia-smi -L || true

MODELS="released_clemson released_oxwalk clemson combined all3 oxwalk"
TAG="${LIMIT:+_smoke}"
for f in runs/released/clemson/fold_15/model.pt runs/released/oxwalk/fold_10/bin_ranges.csv; do
  [ -f "$f" ] || { echo "ERROR: $f missing - run python scripts/prepare_released_weights.py first" >&2; exit 1; }
done

case "$SLURM_ARRAY_TASK_ID" in
  0)
    if [ -n "${PARITY_STRICT:-}" ]; then
      python tests/check_released_parity.py --device cuda:0
    else
      python tests/check_released_parity.py --device cuda:0 || echo "PARITY CHECK REPORTED MISMATCHES (see above; analysis/cache/released_parity.csv)"
    fi ;;
  1)
    [ -z "${LIMIT:-}" ] || echo "note: marea_eval has no --limit; scoring all 68 MAREA recordings"
    python analysis/marea_eval.py --only marea --force --models $MODELS --tag "_released${TAG}" ;;
  2)
    python analysis/orient_tvs.py --device cuda:0 --models $MODELS --tag "_released${TAG}" ${LIMIT:+--limit $LIMIT} ;;
  3)
    [ -f data/mobig/manifest.csv ] || { echo "ERROR: data/mobig/manifest.csv missing - run python -I scripts/prepare_mobig.py" >&2; exit 1; }
    python analysis/mobig_eval.py --device cuda:0 --tag "${TAG}" ${LIMIT:+--limit $LIMIT} ;;
  *)
    echo "unknown task $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac
