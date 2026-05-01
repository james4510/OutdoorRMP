# Radio Map Prediction-based Shadow Region Detection in Sub-7 GHz Environments

This repository contains the core evaluation scripts and selected result figures
for a comparative study of CNN-based RadioUNet and Transformer-based TransPathNet
architectures for shadow region detection via deep learning-based Radio Map Prediction.

## Main models

- RadioUNet-S (35M)
- RadioUNet-M (97M)
- TransPathNet-S (38M)
- TransPathNet-B (111M)

## Main evaluation setup

- Frequencies: 1750, 2750, 3750, 4750, 5750 MHz
- Representative analysis frequencies: 1750, 4750, 5750 MHz
- Fixed shadow thresholds: -60, -65, -70 dBm
- Main representative threshold: -65 dBm
- Boundary region: pixels satisfying |GT RSS - threshold| <= 10 dB

## Repository structure

```text
scripts/
  final_eval_metrics_fixed_threshold_heatmap.py
  final_samplewise_cdf_v3.py
  boundary_only_cdf_analysis.py
  boundary_only_cdf_analysis_x10.py
  final_qualitative_split_allpairs_v11.py
  run_1750_4750_5750_shared_pipeline.sh
  qualitative_shared_sample_1750_4750_5750.py

results/
  fixed_threshold_metrics/
  samplewise_cdf/
  boundary_cdf/
  qualitative_3750/
  representative_1750_4750_5750/
```

## Reproducing representative results

The dataset and checkpoints are not included in this repository.

Expected local paths:

```text
data/
model_checkpoints/
```

Run:

```bash
bash scripts/run_1750_4750_5750_shared_pipeline.sh
```

## Notes

- `model_checkpoints/` and raw dataset files are excluded from this repository.
- CSV files included under `results/` are selected outputs for reproducibility and visualization.
- Figures are selected representative results used for paper and lab meeting discussion.
