#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-python}"
CLEMSON_ROOT="${CLEMSON_DATA_ROOT:-$ROOT/data/clemson}"
OXWALK_15HZ_ROOT="${OXWALK_15HZ_ROOT:-$ROOT/data/oxwalk_wrist_15hz}"
OUT_DIR="${OUTPUT_DIR:-$ROOT/runs/cross_dataset}"
LOG_DIR="$OUT_DIR/_logs"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"

# OxWalk is natively 25Hz; run the resampling step once first to produce
# $OXWALK_15HZ_ROOT (see step_counting/cross_dataset/preprocessing/
# resample_oxwalk_25to15hz.py --src <25Hz wrist dir> --dst "$OXWALK_15HZ_ROOT").

"$PY" step_counting/cross_dataset/code/train_cross_dataset.py \
  --train_dataset oxwalk --test_dataset clemson \
  --clemson_root "$CLEMSON_ROOT" --oxwalk_root "$OXWALK_15HZ_ROOT" --output_dir "$OUT_DIR" \
  --exp_name train_oxwalk_test_clemson --device cuda:0 \
  >"$LOG_DIR/train_oxwalk_test_clemson.log" 2>&1 &

"$PY" step_counting/cross_dataset/code/train_cross_dataset.py \
  --train_dataset clemson --test_dataset oxwalk \
  --clemson_root "$CLEMSON_ROOT" --oxwalk_root "$OXWALK_15HZ_ROOT" --output_dir "$OUT_DIR" \
  --exp_name train_clemson_test_oxwalk --device cuda:1 \
  >"$LOG_DIR/train_clemson_test_oxwalk.log" 2>&1 &

wait
echo "[$(date '+%F %T')] Both cross-dataset runs finished."
