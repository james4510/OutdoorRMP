#!/usr/bin/env bash
set -euo pipefail

# Repeated random-split holdout experiments for reviewer response.
#
# Quick dry run:
#   DRY_RUN=1 bash scripts/run_multi_seed_holdout_experiments.sh
#
# Small smoke run:
#   SEEDS="0" FREQS="4750" MODELS="radiounet-small" EPOCHS=1 \
#     bash scripts/run_multi_seed_holdout_experiments.sh
#
# Full representative run:
#   bash scripts/run_multi_seed_holdout_experiments.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-codex}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$MPLCONFIGDIR"

ROOT="${ROOT:-$PROJECT_ROOT/data}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/results_multi_seed_holdout}"

SEEDS="${SEEDS:-42 43 44 45 46}"
FREQS="${FREQS:-1750 4750 5750}"
MODELS="${MODELS:-radiounet-small radiounet-medium transpathnet-small transpathnet-base}"
THRESHOLDS="${THRESHOLDS:--70 -65 -60}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
AMP="${AMP:-0}"
TRANSPATHNET_BATCH_SIZE="${TRANSPATHNET_BATCH_SIZE:-$BATCH_SIZE}"
TRANSPATHNET_ACCUM_STEPS="${TRANSPATHNET_ACCUM_STEPS:-$ACCUM_STEPS}"
TRANSPATHNET_AMP="${TRANSPATHNET_AMP:-$AMP}"
LR="${LR:-1e-4}"
IMG_SIZE="${IMG_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"

TRAIN="${TRAIN:-1}"
EVAL="${EVAL:-1}"
AGGREGATE="${AGGREGATE:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DETERMINISTIC="${DETERMINISTIC:-0}"
DRY_RUN="${DRY_RUN:-0}"

EVAL_SCRIPT="$SCRIPT_DIR/final_eval_metrics_fixed_threshold_heatmap.py"
AGG_SCRIPT="$SCRIPT_DIR/aggregate_seed_metrics.py"

read -r -a SEED_ARR <<< "$SEEDS"
read -r -a FREQ_ARR <<< "$FREQS"
read -r -a MODEL_ARR <<< "$MODELS"
read -r -a THRESHOLD_ARR <<< "$THRESHOLDS"

run_cmd() {
  printf '>'
  printf ' %q' "$@"
  printf '\n'
  if [ "$DRY_RUN" != "1" ]; then
    "$@"
  fi
}

model_to_arch_size() {
  case "$1" in
    radiounet-small)
      echo "radiounet small"
      ;;
    radiounet-medium)
      echo "radiounet medium"
      ;;
    transpathnet-small)
      echo "transpathnet small"
      ;;
    transpathnet-base)
      echo "transpathnet base"
      ;;
    *)
      echo "[ERROR] Unknown model tag: $1" >&2
      exit 1
      ;;
  esac
}

echo "============================================================"
echo "Repeated holdout experiment"
echo "============================================================"
echo "ROOT        : $ROOT"
echo "OUT_DIR     : $OUT_DIR"
echo "SEEDS       : $SEEDS"
echo "FREQS       : $FREQS"
echo "MODELS      : $MODELS"
echo "THRESHOLDS  : $THRESHOLDS"
echo "EPOCHS      : $EPOCHS"
echo "BATCH/ACCUM : $BATCH_SIZE / $ACCUM_STEPS"
echo "AMP         : $AMP"
echo "TPN B/ACCUM : $TRANSPATHNET_BATCH_SIZE / $TRANSPATHNET_ACCUM_STEPS"
echo "TPN AMP     : $TRANSPATHNET_AMP"
echo "TRAIN/EVAL  : $TRAIN / $EVAL"
echo "DRY_RUN     : $DRY_RUN"
echo "============================================================"

mkdir -p "$OUT_DIR"

if [ "$TRAIN" = "1" ]; then
  for seed in "${SEED_ARR[@]}"; do
    ckpt_dir="$OUT_DIR/seed_${seed}/model_checkpoints"
    mkdir -p "$ckpt_dir"

    for freq in "${FREQ_ARR[@]}"; do
      for tag in "${MODEL_ARR[@]}"; do
        read -r arch size <<< "$(model_to_arch_size "$tag")"
        ckpt_tag="${tag//-/_}"
        save_dir="$ckpt_dir/checkpoints_${ckpt_tag}_${freq}"
        best_path="$save_dir/best.pth"
        complete_marker="$save_dir/training_complete.txt"
        training_curve="$save_dir/training_curve.png"
        train_batch_size="$BATCH_SIZE"
        train_accum_steps="$ACCUM_STEPS"
        train_amp="$AMP"
        if [[ "$tag" == transpathnet-* ]]; then
          train_batch_size="$TRANSPATHNET_BATCH_SIZE"
          train_accum_steps="$TRANSPATHNET_ACCUM_STEPS"
          train_amp="$TRANSPATHNET_AMP"
        fi

        if [ "$SKIP_EXISTING" = "1" ] && [ -f "$best_path" ] && { [ -f "$complete_marker" ] || [ -f "$training_curve" ]; }; then
          echo "[SKIP] Existing completed checkpoint: $best_path"
          continue
        fi

        echo "------------------------------------------------------------"
        echo "[Train] seed=$seed freq=${freq}MHz model=$tag"
        echo "------------------------------------------------------------"

        cmd=(
          "$PYTHON" -m ikeee.bratlab_train
          --mode train
          --root "$ROOT"
          --freq "$freq"
          --model "$arch"
          --size "$size"
          --img_size "$IMG_SIZE"
          --batch_size "$train_batch_size"
          --accum_steps "$train_accum_steps"
          --epochs "$EPOCHS"
          --lr "$LR"
          --num_workers "$NUM_WORKERS"
          --save_dir "$save_dir"
          --seed "$seed"
        )
        if [ "$DETERMINISTIC" = "1" ]; then
          cmd+=(--deterministic)
        fi
        if [ "$train_amp" = "1" ]; then
          cmd+=(--amp)
        fi
        run_cmd "${cmd[@]}"
      done
    done
  done
fi

if [ "$EVAL" = "1" ]; then
  for seed in "${SEED_ARR[@]}"; do
    ckpt_dir="$OUT_DIR/seed_${seed}/model_checkpoints"
    metrics_dir="$OUT_DIR/seed_${seed}/metrics"
    mkdir -p "$metrics_dir"

    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$metrics_dir/summary_metrics_fixed_threshold.csv" ]; then
      echo "[SKIP] Existing metrics: $metrics_dir/summary_metrics_fixed_threshold.csv"
      continue
    fi

    echo "------------------------------------------------------------"
    echo "[Eval] seed=$seed"
    echo "------------------------------------------------------------"

    run_cmd \
      "$PYTHON" "$EVAL_SCRIPT" \
      --root "$ROOT" \
      --ckpt_dir "$ckpt_dir" \
      --result_dir "$metrics_dir" \
      --freqs "${FREQ_ARR[@]}" \
      --thresholds "${THRESHOLD_ARR[@]}" \
      --models "${MODEL_ARR[@]}" \
      --img_size "$IMG_SIZE" \
      --num_workers "$NUM_WORKERS" \
      --seed "$seed"
  done
fi

if [ "$AGGREGATE" = "1" ]; then
  echo "------------------------------------------------------------"
  echo "[Aggregate]"
  echo "------------------------------------------------------------"
  run_cmd \
    "$PYTHON" "$AGG_SCRIPT" \
    --result_root "$OUT_DIR" \
    --output_dir "$OUT_DIR/aggregate"
fi

echo "============================================================"
echo "Done. Results root: $OUT_DIR"
echo "Aggregate table: $OUT_DIR/aggregate/summary_metrics_across_seeds.csv"
echo "Paper tables:    $OUT_DIR/aggregate/paper_table_th*.csv"
echo "============================================================"
