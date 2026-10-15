# Radio Map Prediction-based Shadow Region Detection in Sub-7 GHz Environments

English | [한국어](README.ko.md)

This repository contains the dataset loader, model code, training code, evaluation scripts
and selected result figures for a comparative study of CNN-based RadioUNet and
Transformer-based TransPathNet architectures for shadow region detection via deep
learning-based Radio Map Prediction.

## Main models

| Model | Tag | Parameters |
|---|---|---|
| RadioUNet-S | `radiounet-small` | 34.92M |
| RadioUNet-M | `radiounet-medium` | 96.98M |
| TransPathNet-S | `transpathnet-small` | 38.09M |
| TransPathNet-B | `transpathnet-base` | 111.44M |

- Two-stage models: a coarse prediction followed by a fine prediction
- Input: 2 channels (building mask, Gaussian-smoothed position map of the 3 transmitters)
- Output: received power map normalized from [-90, 0] dBm to [0, 1]
- Input and output size: 256 x 256

## Main evaluation setup

- Frequencies: 1750, 2750, 3750, 4750, 5750 MHz
- Representative analysis frequencies: 1750, 4750, 5750 MHz
- Fixed shadow thresholds: -60, -65, -70 dBm
- Main representative threshold: -65 dBm
- Boundary region: pixels satisfying |GT RSS - threshold| <= 10 dB

## Repository structure

```text
Dockerfile                    runtime image (CUDA 11.8, PyTorch 2.2.0)
docker-compose.yml            container settings (rmp: GPU, rmp-cpu: CPU)
docker/
  requirements.txt            packages installed on top of the base image
  setup_env.sh                creates .env (user IDs, data and checkpoint paths)
docs/
  [OutdoorRMP]설치메뉴얼.docx   installation manual (Korean)

ikeee/                        dataset, models and training (Python package)
  bratlab_dataset.py          BRAT-Lab dataset, 70/10/20 split with a fixed seed
  bratlab_train.py            training and single-checkpoint evaluation
  model/
    radiounet.py              RadioUNet-S / M
    transpathnet_outdoor.py   TransPathNet-S / B (outdoor version, 2-channel input)
    transnext_backbone.py     lightweight TransNeXt backbone for TransPathNet
    emcad_decoder.py          EMCAD decoder for TransPathNet

scripts/
  check_setup.py                                  checks packages, GPU, dataset and checkpoints
  run_multi_seed_holdout_experiments.sh           training, evaluation and aggregation over seeds
  resume_seed43_44_remaining.sh                   resume script for the seed 43/44 runs
  aggregate_seed_metrics.py                       mean and std across seeds
  final_eval_metrics_fixed_threshold_heatmap.py   fixed-threshold metrics, heatmaps and tables
  boundary_only_cdf_analysis.py                   boundary-only error and miss rate CDFs
  boundary_only_cdf_analysis_x10.py               same as above, error axis 0-10 dB
  final_qualitative_split_allpairs_v11.py         error map and shadow detection figures (one frequency)
  qualitative_shared_sample_1750_4750_5750.py     error map and shadow detection figures (one sample, 3 frequencies)
  run_1750_4750_5750_shared_pipeline.sh           pipeline for the representative results
  final_samplewise_cdf_v3.py                      sample-wise miss rate CDF (CSV input)
  paper_missrate_cdf_builder.py                   miss rate CDF figures and summary table (CSV input)
  plot_existing_cdf_font_plus10.py                CDF figures with larger fonts (CSV input)
  plot_fixed_threshold_heatmap_font_plus10.py     heatmap with larger fonts (CSV input)
  plot_rmse_vs_frequency_paper.py                 RMSE vs. frequency figure (CSV input)

results/
  fixed_threshold_metrics/
  samplewise_cdf/
  boundary_cdf/
  qualitative_3750/
  representative_1750_4750_5750/
```

## Data and checkpoints

The dataset and checkpoints are not included in this repository.
Default locations:

```text
data/
  stations_position.txt                       Tx positions per sample (row1 col1 row2 col2 row3 col3)
  buildings_position/buildings_{n}.mat
  receivedpower_{freq}MHz_mat/TOTAL_REC_{freq}MHz_{n}.mat
model_checkpoints/
  checkpoints_{model}_{size}_{freq}/best.pth   e.g. checkpoints_transpathnet_base_4750/best.pth
```

- n = 1, ..., 2000
- Invalid samples (NaN values, smaller than 50 x 50 pixels) are excluded: 12 samples at 4750 MHz
- Test set: 400 samples (399 at 4750 MHz)

## Installation with Docker

A step-by-step guide in Korean is in `docs/[OutdoorRMP]설치메뉴얼.docx`.

- Requirements: Docker Engine, Docker Compose v2, NVIDIA driver and NVIDIA Container Toolkit
- Image: `outdoor-rmp:latest` (CUDA 11.8, PyTorch 2.2.0)
- The repository is mounted at `/workspace`, so outputs are written to the repository folder on the host.

```bash
git clone https://github.com/james4510/OutdoorRMP.git
cd OutdoorRMP
docker compose build
bash docker/setup_env.sh /path/to/data /path/to/model_checkpoints   # creates .env
docker compose run --rm rmp python scripts/check_setup.py           # should print "Result: OK"
docker compose run --rm rmp bash scripts/run_1750_4750_5750_shared_pipeline.sh
```

- Every command in this README can be run in the container as `docker compose run --rm rmp <command>`.
- `rmp-cpu` runs the same image without a GPU (evaluation and plotting).
- Windows: install Docker Desktop (WSL 2 backend) and run the commands in a WSL (Ubuntu) terminal.

## Training

Run from the repository root.

```bash
python -m ikeee.bratlab_train --mode train \
    --root data --freq 4750 --model radiounet --size small \
    --batch_size 8 --epochs 100 --lr 1e-4 \
    --save_dir model_checkpoints/checkpoints_radiounet_small_4750
```

- Models: `--model radiounet --size small|medium`, `--model transpathnet --size small|base`
- Optimizer: AdamW (weight decay 1e-4), cosine annealing (eta_min 1e-6), gradient clipping 1.0
- Loss: 0.6 * MSE + 0.3 * (1 - SSIM) + 0.1 * MSE(aux) for the fine output, plus 0.4 times the same loss for the coarse output
- `best.pth`: checkpoint with the lowest validation loss

Training settings of the seed-42 checkpoints used for `results/`:

| Model | Batch size | Learning rate | Epochs |
|---|---|---|---|
| RadioUNet-S | 8 | 1e-4 | 100 |
| RadioUNet-M | 4 | 1e-4 | 100 |
| TransPathNet-S | 2 | 1e-4 | 100 |
| TransPathNet-B | 2 | 1e-4 | 100 |

- RadioUNet-S at 4750 MHz: the evaluation scripts use `checkpoints_radiounet_small_4750_v2/best.pth` (lr 3e-5) if it exists

### Single-checkpoint evaluation

RMSE, MAE and bias (dB) over non-building pixels.

```bash
python -m ikeee.bratlab_train --mode eval \
    --root data --freq 4750 --model radiounet --size small \
    --ckpt model_checkpoints/checkpoints_radiounet_small_4750/best.pth \
    --result_dir runs/eval_radiounet_small_4750
```

- The default `--result_dir` is `./results`. Use another directory to keep `results/` unchanged.

### Repeated holdout with different split seeds

```bash
DRY_RUN=1 bash scripts/run_multi_seed_holdout_experiments.sh   # print commands only
bash scripts/run_multi_seed_holdout_experiments.sh             # seeds 42-46, 1750/4750/5750 MHz
```

- Settings can be changed with environment variables (`SEEDS`, `FREQS`, `MODELS`, `ROOT`, `OUT_DIR`, `EPOCHS`, etc.)
- Output directory: `results_multi_seed_holdout/`
- Seeds 43 and 44: TransPathNet with `TRANSPATHNET_BATCH_SIZE=2 TRANSPATHNET_ACCUM_STEPS=4 TRANSPATHNET_AMP=1`, RadioUNet with batch size 8 without AMP

## Reproducing representative results

```bash
bash scripts/run_1750_4750_5750_shared_pipeline.sh

# data and checkpoints in other directories
ROOT=/path/to/data CKPT_DIR=/path/to/model_checkpoints \
    bash scripts/run_1750_4750_5750_shared_pipeline.sh
```

- Output directory: `results_1750_4750_5750_regenerated/` (set `OUT_DIR` to change it)

## Result files and scripts

| Result | Script |
|---|---|
| `results/fixed_threshold_metrics/heatmap_fixed_threshold_th-65.png`, `*_metrics_fixed_threshold.csv` | `final_eval_metrics_fixed_threshold_heatmap.py` |
| `results/fixed_threshold_th-65.png`, `results/fixed_threshold_metrics/heatmap_fixed_threshold_th-65_font_plus10.png` | `plot_fixed_threshold_heatmap_font_plus10.py` |
| `results/samplewise_cdf/fig_cdf_missrate_*.png` | `final_samplewise_cdf_v3.py` |
| `results/samplewise_cdf/samplewise_miss_rate_rows.csv`, `cdf_summary_table.csv` | `paper_missrate_cdf_builder.py` |
| `results/samplewise_cdf/samplewise_error_rates.csv` | earlier script (see Notes) |
| `results/boundary_cdf/fig_boundary_{error,combined}_cdf_th-65_bw10.png` | `boundary_only_cdf_analysis.py` |
| other files in `results/boundary_cdf/` | `boundary_only_cdf_analysis_x10.py` |
| `results/qualitative_3750/` | `final_qualitative_split_allpairs_v11.py` (3750 MHz, sample 1486) |
| `results/representative_1750_4750_5750/{metrics,miss_cdf,boundary_cdf,qualitative}/` | `run_1750_4750_5750_shared_pipeline.sh` (`final_eval_metrics_fixed_threshold_heatmap.py`, `final_samplewise_cdf_v3.py`, `boundary_only_cdf_analysis_x10.py`, `qualitative_shared_sample_1750_4750_5750.py`) |
| `results/representative_1750_4750_5750/*_font_plus10/` | `plot_existing_cdf_font_plus10.py` |

- All checkpoint-based results are computed on the same test samples (seed 42 split).

## Notes

- `model_checkpoints/` and raw dataset files are excluded from this repository.
- CSV files included under `results/` are selected outputs for reproducibility and visualization.
- Figures are selected representative results used for paper and lab meeting discussion.
- Some figures were generated with earlier versions of the plotting scripts. Running the current scripts may give different font sizes or layouts.
- `results/samplewise_cdf/samplewise_error_rates.csv` was generated by an earlier script with frequency-dependent thresholds (1750: -61.7, 2750: -64.9, 3750: -67.4, 4750: -69.5, 5750: -71.1 dBm). The script is not available. `final_samplewise_cdf_v3.py` is the fixed-threshold version.
- `plot_rmse_vs_frequency_paper.py` reads `summary_metrics_long.csv` of the 5-frequency evaluation by default, which is not included. Set `--summary_csv` and `--out` when running it. Any summary CSV with `rmse_dbm_mean` and `rmse_dbm_std` columns works, e.g. `results/fixed_threshold_metrics/summary_metrics_fixed_threshold.csv`.

## References

The model code is based on the following works.

- RadioUNet: R. Levie et al., "RadioUNet: Fast Radio Map Estimation With Convolutional Neural Networks," IEEE Transactions on Wireless Communications, 2021 (arXiv:1911.09002)
- TransPathNet: https://github.com/LiXin97/TransPathNet
- TransNeXt: https://github.com/DaiShiResearch/TransNeXt (Apache-2.0)
- EMCAD: Rahman et al., CVPR 2024, https://github.com/SLDGroup/EMCAD
