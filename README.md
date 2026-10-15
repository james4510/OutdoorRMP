# Radio Map Prediction-based Shadow Region Detection in Sub-7 GHz Environments

This repository contains the dataset loader, model definitions, training code,
evaluation scripts and selected result figures for a comparative study of
CNN-based RadioUNet and Transformer-based TransPathNet architectures for shadow
region detection via deep learning-based Radio Map Prediction.

## Main models

| Model | Tag | Parameters |
|---|---|---|
| RadioUNet-S | `radiounet-small` | 34.92M |
| RadioUNet-M | `radiounet-medium` | 96.98M |
| TransPathNet-S | `transpathnet-small` | 38.09M |
| TransPathNet-B | `transpathnet-base` | 111.44M |

Both architectures are two-stage (coarse → fine) networks. The input has two
channels (building mask, Gaussian-smoothed map of the three Tx positions) and the
target is the received-power map normalized from [-90, 0] dBm to [0, 1], at 256×256.

## Main evaluation setup

- Frequencies: 1750, 2750, 3750, 4750, 5750 MHz
- Representative analysis frequencies: 1750, 4750, 5750 MHz
- Fixed shadow thresholds: -60, -65, -70 dBm
- Main representative threshold: -65 dBm
- Boundary region: pixels satisfying |GT RSS - threshold| <= 10 dB

## Repository structure

```text
ikeee/                          # Python package: dataset, models, training
  bratlab_dataset.py            # BRAT-Lab dataset and fixed 70/10/20 split dataloaders
  bratlab_train.py              # training / single-checkpoint evaluation
  model/
    radiounet.py                # RadioUNet-S / -M
    transpathnet_outdoor.py     # TransPathNet-S / -B (outdoor, 2-channel input)
    transnext_backbone.py       # TransNeXt-style backbone used by TransPathNet
    emcad_decoder.py            # EMCAD decoder used by TransPathNet

scripts/
  # training
  run_multi_seed_holdout_experiments.sh   # train -> evaluate -> aggregate over split seeds
  resume_seed43_44_remaining.sh           # resume helper used for the seed 43/44 runs
  aggregate_seed_metrics.py               # mean / std across seeds
  # evaluation (needs data and checkpoints)
  final_eval_metrics_fixed_threshold_heatmap.py
  boundary_only_cdf_analysis.py
  boundary_only_cdf_analysis_x10.py
  final_qualitative_split_allpairs_v11.py
  qualitative_shared_sample_1750_4750_5750.py
  run_1750_4750_5750_shared_pipeline.sh
  # plotting from existing CSV outputs
  final_samplewise_cdf_v3.py
  paper_missrate_cdf_builder.py
  plot_existing_cdf_font_plus10.py
  plot_fixed_threshold_heatmap_font_plus10.py
  plot_rmse_vs_frequency_paper.py

results/
  fixed_threshold_metrics/
  samplewise_cdf/
  boundary_cdf/
  qualitative_3750/
  representative_1750_4750_5750/
```

## Data and checkpoints

The dataset and checkpoints are not included in this repository.
By default the scripts expect them at the repository root:

```text
data/
  stations_position.txt                        # line n: row1 col1 row2 col2 row3 col3 (3 Tx of sample n)
  buildings_position/buildings_{n}.mat
  receivedpower_{freq}MHz_mat/TOTAL_REC_{freq}MHz_{n}.mat
model_checkpoints/
  checkpoints_{model}_{size}_{freq}/best.pth    # e.g. checkpoints_transpathnet_base_4750/best.pth
```

`n` runs from 1 to 2000. Samples that contain NaN or are smaller than 50×50 pixels are
skipped (12 at 4750 MHz), so the test split has 400 samples (399 at 4750 MHz).

## Training

Run from the repository root so that the `ikeee` package can be imported:

```bash
python -m ikeee.bratlab_train --mode train \
    --root data --freq 4750 --model radiounet --size small \
    --batch_size 8 --epochs 100 --lr 1e-4 \
    --save_dir model_checkpoints/checkpoints_radiounet_small_4750
```

Use `--model radiounet --size small|medium` or `--model transpathnet --size small|base`.
Training uses AdamW (weight decay 1e-4), cosine annealing (eta_min 1e-6), gradient
clipping at 1.0, and the loss 0.6·MSE + 0.3·(1 − SSIM) + 0.1·MSE(aux) on the fine output
plus 0.4 × the same loss on the coarse output. `best.pth` is the epoch with the lowest
validation loss.

Settings stored in the seed-42 checkpoints behind the results in `results/`:

| Model | Batch size | Learning rate | Epochs |
|---|---|---|---|
| RadioUNet-S | 8 | 1e-4 | 100 |
| RadioUNet-M | 4 | 1e-4 | 100 |
| TransPathNet-S | 2 | 1e-4 | 100 |
| TransPathNet-B | 2 | 1e-4 | 100 |

For RadioUNet-S at 4750 MHz the evaluation scripts load
`checkpoints_radiounet_small_4750_v2/best.pth` (retrained with lr 3e-5) when it exists.

To evaluate a single checkpoint (RMSE / MAE / bias in dB on non-building pixels, plus example plots):

```bash
python -m ikeee.bratlab_train --mode eval \
    --root data --freq 4750 --model radiounet --size small \
    --ckpt model_checkpoints/checkpoints_radiounet_small_4750/best.pth \
    --result_dir runs/eval_radiounet_small_4750
```

The default `--result_dir` is `./results`, so pass another directory to keep the
tracked `results/` folder unchanged.

### Repeated random-split holdout

```bash
DRY_RUN=1 bash scripts/run_multi_seed_holdout_experiments.sh   # print the commands only
bash scripts/run_multi_seed_holdout_experiments.sh             # seeds 42-46, 1750/4750/5750 MHz
```

Every setting can be overridden from the environment (`SEEDS`, `FREQS`, `MODELS`, `ROOT`,
`OUT_DIR`, `EPOCHS`, ...). Outputs go to `results_multi_seed_holdout/`. The seed 43/44 runs
used `TRANSPATHNET_BATCH_SIZE=2 TRANSPATHNET_ACCUM_STEPS=4 TRANSPATHNET_AMP=1`
(RadioUNet: batch size 8, no AMP).

## Reproducing representative results

```bash
bash scripts/run_1750_4750_5750_shared_pipeline.sh

# data / checkpoints stored elsewhere
ROOT=/path/to/data CKPT_DIR=/path/to/model_checkpoints \
    bash scripts/run_1750_4750_5750_shared_pipeline.sh
```

Outputs are written to `results_1750_4750_5750_regenerated/` (override with `OUT_DIR`).

## Which script produced which result

| Result | Script |
|---|---|
| `results/fixed_threshold_metrics/heatmap_fixed_threshold_th-65.png`, `*_metrics_fixed_threshold.csv` | `final_eval_metrics_fixed_threshold_heatmap.py` |
| `results/fixed_threshold_th-65.png`, `results/fixed_threshold_metrics/heatmap_fixed_threshold_th-65_font_plus10.png` | `plot_fixed_threshold_heatmap_font_plus10.py` |
| `results/samplewise_cdf/fig_cdf_missrate_*.png` | `final_samplewise_cdf_v3.py` |
| `results/samplewise_cdf/samplewise_miss_rate_rows.csv`, `cdf_summary_table.csv` | `paper_missrate_cdf_builder.py` |
| `results/samplewise_cdf/samplewise_error_rates.csv` | legacy script, see Notes |
| `results/boundary_cdf/fig_boundary_{error,combined}_cdf_th-65_bw10.png` | `boundary_only_cdf_analysis.py` |
| other files in `results/boundary_cdf/` | `boundary_only_cdf_analysis_x10.py` |
| `results/qualitative_3750/` | `final_qualitative_split_allpairs_v11.py` (3750 MHz, sample 1486) |
| `results/representative_1750_4750_5750/{metrics,miss_cdf,boundary_cdf,qualitative}/` | `run_1750_4750_5750_shared_pipeline.sh` (`final_eval_metrics_fixed_threshold_heatmap.py`, `final_samplewise_cdf_v3.py`, `boundary_only_cdf_analysis_x10.py`, `qualitative_shared_sample_1750_4750_5750.py`) |
| `results/representative_1750_4750_5750/*_font_plus10/` | `plot_existing_cdf_font_plus10.py` |

All checkpoint-based results use the seed-42 70/10/20 split, i.e. the same test samples.

## Notes

- `model_checkpoints/` and raw dataset files are excluded from this repository.
- CSV files included under `results/` are selected outputs for reproducibility and visualization.
- Figures are selected representative results used for paper and lab meeting discussion.
  Some were made with earlier revisions of the plotting code, so re-running the current
  scripts can change fonts or layout.
- `results/samplewise_cdf/samplewise_error_rates.csv` comes from an earlier sample-wise CDF
  script that used frequency-dependent thresholds (1750: -61.7, 2750: -64.9, 3750: -67.4,
  4750: -69.5, 5750: -71.1 dBm). That script is no longer available;
  `final_samplewise_cdf_v3.py` is its fixed-threshold replacement.
- `plot_rmse_vs_frequency_paper.py` defaults to the 5-frequency `summary_metrics_long.csv` of the
  original workspace, which is not included. Pass `--summary_csv` (any summary CSV with
  `rmse_dbm_mean` / `rmse_dbm_std` columns, e.g.
  `results/fixed_threshold_metrics/summary_metrics_fixed_threshold.csv`) and `--out`.

## References

The model code re-implements or adapts:

- RadioUNet: R. Levie et al., "RadioUNet: Fast Radio Map Estimation With Convolutional Neural
  Networks," IEEE Transactions on Wireless Communications, 2021 (arXiv:1911.09002)
- TransPathNet: https://github.com/LiXin97/TransPathNet
- TransNeXt: https://github.com/DaiShiResearch/TransNeXt (Apache-2.0)
- EMCAD: Rahman et al., CVPR 2024, https://github.com/SLDGroup/EMCAD
