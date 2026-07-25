#!/usr/bin/env bash
# Reproduces the OxWalk wrist U-Net experiments corresponding to
# manuscript Table 2 (loss objective comparison, OxWalk column).
# All four configs share the same fixed preprocessing/representation
# setting (unprocessed raster input, motion-energy realigned training
# labels, original annotations for validation/test) and only vary the
# loss objective, matching Section 4.3.2 of the manuscript.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-python}"
DATA_ROOT="${OXWALK_REALIGNED_WRIST_ROOT:-$ROOT/data/oxwalk/OxWalk_Dec2022_peakvalley_025s/Wrist_25Hz}"
TEST_ANN_ROOT="${OXWALK_ORIGINAL_WRIST_ROOT:-$ROOT/data/oxwalk/OxWalk_Dec2022/Wrist_25Hz}"
OUT_DIR="${OUTPUT_DIR:-$ROOT/runs/oxwalk}"
LOG_DIR="$OUT_DIR/_logs"
mkdir -p "$LOG_DIR"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"

COMMON_ARGS=(
  --data_root "$DATA_ROOT"
  --test_annotation_root "$TEST_ANN_ROOT"
  --output_dir "$OUT_DIR"
  --conditions Regular Irregular SemiRegular
  --epochs 120
  --batch_size 64
  --sampling_rate 25
  --window_sec 60
  --height_bins 16
  --overlap_ratio 0.0
  --num_workers 4
  --min_steps_for_walk 1
  --count_mode mass
  --eval_count_mode mass
  --label_width 7
  --label_mode centered
  --preprocess_mode none
  --switch_threshold 0.5
  --switch_min_run 1
  --lr 0.0003
  --weight_decay 0.0001
  --eta_min 0.00001
  --ramp_epochs 50
  --patience 8
  --min_delta 0.00001
  --model_type dense
  --alpha_max 0.15
)

run_config() {
  local exp_name="$1" device="$2" loss_type="$3" warmup_ratio="$4"
  local log="$LOG_DIR/${exp_name}.log"
  echo "[$(date '+%F %T')] START $exp_name on $device (loss=$loss_type warmup=$warmup_ratio)"
  "$PY" step_counting/oxwalk/code/main.py \
    "${COMMON_ARGS[@]}" \
    --exp_name "$exp_name" \
    --device "$device" \
    --loss_type "$loss_type" \
    --warmup_ratio "$warmup_ratio" \
    >"$log" 2>&1 &
}

# Table 2 "U-Net BCE + count loss" (scheduled hybrid) -- main proposed model, OxWalk 12.12% MAPE result.
run_config oxwalk_wrist_hybrid_scheduled cuda:0 hybrid_bce 0.25
# Table 2 "U-Net BCE only".
run_config oxwalk_wrist_bce_only         cuda:1 bce        0.25
# Table 2 "U-Net count loss only" (implemented as dense-MSE objective in the codebase).
run_config oxwalk_wrist_count_only       cuda:0 mse        0.25
# Table 2 "U-Net BCE + count loss, no schedule".
run_config oxwalk_wrist_hybrid_nosched   cuda:1 hybrid_bce 0.0

wait
echo "[$(date '+%F %T')] All OxWalk configs finished."
