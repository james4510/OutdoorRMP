#!/usr/bin/env bash
set -e

# ======================================================
# Representative frequency pipeline
# Frequencies: 1750 / 4750 / 5750 MHz
#
# Generates:
# 1) Fixed-threshold metrics
# 2) Sample-wise Miss Rate CDF
# 3) Boundary-only Miss Rate CDF
# 4) Boundary-only RSS Error CDF
# 5) Shadow Region Detection Comparison with TP/FP/FN
# 6) Error Map Comparison
#
# The qualitative figures use ONE shared sample id across all frequencies.
# If SAMPLE_ID="auto", the script selects a common sample automatically.
# ======================================================

# --------- Edit only these if needed ---------
ROOT="/home/james4510/OutdoorRadioMapPrediction/data"
CKPT_DIR="./model_checkpoints"
OUT_DIR="./results_1750_4750_5750"

FREQS="1750 4750 5750"
THRESHOLDS="-70 -65 -60"

# Representative threshold for qualitative figures and sample selection.
QUAL_THRESHOLD=-65

# Use "auto" to find a common sample id across 1750/4750/5750 MHz.
SAMPLE_ID="auto"

BOUNDARY_WIDTH=10
NUM_WORKERS=4
# --------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REQUIRED=(
  "final_eval_metrics_fixed_threshold_heatmap.py"
  "final_samplewise_cdf_v3.py"
  "boundary_only_cdf_analysis_x10.py"
  "qualitative_shared_sample_1750_4750_5750.py"
)

for f in "${REQUIRED[@]}"; do
  if [ ! -f "$SCRIPT_DIR/$f" ]; then
    echo "[ERROR] Missing required script: $SCRIPT_DIR/$f"
    echo "        Put this runner and all required Python files in the same directory."
    exit 1
  fi
done

mkdir -p "$OUT_DIR/metrics" "$OUT_DIR/miss_cdf" "$OUT_DIR/boundary_cdf" "$OUT_DIR/qualitative"

echo "============================================================"
echo "[1/4] Fixed-threshold evaluation metrics"
echo "============================================================"
python "$SCRIPT_DIR/final_eval_metrics_fixed_threshold_heatmap.py" \
  --root "$ROOT" \
  --ckpt_dir "$CKPT_DIR" \
  --result_dir "$OUT_DIR/metrics" \
  --freqs $FREQS \
  --thresholds $THRESHOLDS \
  --num_workers "$NUM_WORKERS"

METRIC_CSV="$OUT_DIR/metrics/per_sample_metrics_fixed_threshold.csv"
if [ ! -f "$METRIC_CSV" ]; then
  echo "[ERROR] Expected metric CSV not found: $METRIC_CSV"
  exit 1
fi

echo "============================================================"
echo "[2/4] Sample-wise Miss Rate CDF"
echo "============================================================"
python "$SCRIPT_DIR/final_samplewise_cdf_v3.py" \
  --input_csv "$METRIC_CSV" \
  --result_dir "$OUT_DIR/miss_cdf" \
  --freqs $FREQS \
  --main_freqs $FREQS \
  --zoom_freqs 1750 \
  --thresholds $THRESHOLDS

echo "============================================================"
echo "[3/4] Boundary-only CDFs"
echo "============================================================"
python "$SCRIPT_DIR/boundary_only_cdf_analysis_x10.py" \
  --root "$ROOT" \
  --ckpt_dir "$CKPT_DIR" \
  --result_dir "$OUT_DIR/boundary_cdf" \
  --freqs $FREQS \
  --threshold_mode fixed \
  --thresholds $THRESHOLDS \
  --boundary_width_dbm "$BOUNDARY_WIDTH" \
  --error_x_max 10 \
  --num_workers "$NUM_WORKERS"

echo "============================================================"
echo "[4/4] Shared-sample qualitative figures"
echo "============================================================"
python "$SCRIPT_DIR/qualitative_shared_sample_1750_4750_5750.py" \
  --root "$ROOT" \
  --ckpt_dir "$CKPT_DIR" \
  --metrics_csv "$METRIC_CSV" \
  --result_dir "$OUT_DIR/qualitative" \
  --freqs $FREQS \
  --sample_id "$SAMPLE_ID" \
  --selection_threshold "$QUAL_THRESHOLD" \
  --shadow_threshold "$QUAL_THRESHOLD" \
  --num_workers "$NUM_WORKERS"

echo ""
echo "============================================================"
echo "Done."
echo "Results saved in: $OUT_DIR"
echo ""
echo "Main output folders:"
echo "  - Metrics:        $OUT_DIR/metrics"
echo "  - Miss CDF:       $OUT_DIR/miss_cdf"
echo "  - Boundary CDF:   $OUT_DIR/boundary_cdf"
echo "  - Qualitative:    $OUT_DIR/qualitative"
echo ""
echo "Selected sample info:"
echo "  $OUT_DIR/qualitative/selected_shared_sample.csv"
echo "============================================================"
