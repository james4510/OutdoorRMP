#!/usr/bin/env bash
set -euo pipefail

# Resume the reviewer-response repeated-seed experiment.
#
# Intended use:
#   1) Reuse the existing seed 42 metrics from results/representative_1750_4750_5750.
#   2) Train only missing checkpoints for seeds 43 and 44.
#   3) Recompute full metrics for seeds 43 and 44.
#   4) Aggregate seed 42/43/44 into manuscript-ready tables.
#
# Run from the repository root:
#   bash scripts/resume_seed43_44_remaining.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/results_multi_seed_holdout}"
SEEDS_TO_RUN="${SEEDS_TO_RUN:-43 44}"

SEED42_METRICS_DIR="${SEED42_METRICS_DIR:-$PROJECT_ROOT/results/representative_1750_4750_5750/metrics}"
SEED42_TARGET_DIR="$OUT_DIR/seed_42/metrics"

RUNNER="$SCRIPT_DIR/run_multi_seed_holdout_experiments.sh"
AGG_SCRIPT="$SCRIPT_DIR/aggregate_seed_metrics.py"

echo "============================================================"
echo "Resume seed 43/44 experiment"
echo "============================================================"
echo "PROJECT_ROOT       : $PROJECT_ROOT"
echo "OUT_DIR            : $OUT_DIR"
echo "SEEDS_TO_RUN       : $SEEDS_TO_RUN"
echo "SEED42_METRICS_DIR : $SEED42_METRICS_DIR"
echo "============================================================"

if [ ! -f "$SEED42_METRICS_DIR/summary_metrics_fixed_threshold.csv" ]; then
  echo "[ERROR] Missing seed 42 summary:"
  echo "        $SEED42_METRICS_DIR/summary_metrics_fixed_threshold.csv"
  exit 1
fi

mkdir -p "$SEED42_TARGET_DIR"
cp "$SEED42_METRICS_DIR/summary_metrics_fixed_threshold.csv" \
   "$SEED42_TARGET_DIR/summary_metrics_fixed_threshold.csv"

if [ -f "$SEED42_METRICS_DIR/per_sample_metrics_fixed_threshold.csv" ]; then
  cp "$SEED42_METRICS_DIR/per_sample_metrics_fixed_threshold.csv" \
     "$SEED42_TARGET_DIR/per_sample_metrics_fixed_threshold.csv"
fi

echo "------------------------------------------------------------"
echo "[1/3] Train remaining checkpoints for seeds: $SEEDS_TO_RUN"
echo "      Existing best.pth files are skipped."
echo "------------------------------------------------------------"
SEEDS="$SEEDS_TO_RUN" \
TRAIN=1 \
EVAL=0 \
AGGREGATE=0 \
SKIP_EXISTING=1 \
bash "$RUNNER"

echo "------------------------------------------------------------"
echo "[2/3] Recompute full metrics for seeds: $SEEDS_TO_RUN"
echo "      Existing metric CSVs are overwritten to avoid partial results."
echo "------------------------------------------------------------"
SEEDS="$SEEDS_TO_RUN" \
TRAIN=0 \
EVAL=1 \
AGGREGATE=0 \
SKIP_EXISTING=0 \
bash "$RUNNER"

echo "------------------------------------------------------------"
echo "[3/3] Aggregate seed 42 plus seeds: $SEEDS_TO_RUN"
echo "------------------------------------------------------------"
"$PYTHON" "$AGG_SCRIPT" \
  --result_root "$OUT_DIR" \
  --output_dir "$OUT_DIR/aggregate"

echo "============================================================"
echo "Done."
echo "Aggregate CSV : $OUT_DIR/aggregate/summary_metrics_across_seeds.csv"
echo "Paper tables  : $OUT_DIR/aggregate/paper_table_th*.csv"
echo "============================================================"
