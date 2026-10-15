# Radio Map Prediction-based Shadow Region Detection in Sub-7 GHz Environments

[English](README.md) | 한국어

- Sub-7 GHz 환경에서 딥러닝 기반 Radio Map Prediction으로 음영 영역을 검출하는 연구의 코드와 결과를 정리한 저장소이다.
- CNN 기반 RadioUNet과 Transformer 기반 TransPathNet의 음영 영역 검출 성능을 비교한다.
- 데이터셋 로더, 모델, 학습 코드, 평가 스크립트, 주요 결과 그림을 포함한다.

## 모델

| 모델 | 태그 | 파라미터 수 |
|---|---|---|
| RadioUNet-S | `radiounet-small` | 34.92M |
| RadioUNet-M | `radiounet-medium` | 96.98M |
| TransPathNet-S | `transpathnet-small` | 38.09M |
| TransPathNet-B | `transpathnet-base` | 111.44M |

- 두 모델 모두 coarse 예측 후 fine 예측을 수행하는 2단계 구조이다.
- 입력: 2채널 (건물 마스크, 송신기 3개의 위치에 Gaussian 필터를 적용한 맵)
- 출력: [-90, 0] dBm 범위의 수신 전력을 [0, 1]로 정규화한 맵
- 입출력 크기: 256 x 256

## 평가 설정

- 주파수: 1750, 2750, 3750, 4750, 5750 MHz
- 대표 분석 주파수: 1750, 4750, 5750 MHz
- 고정 음영 임계값: -60, -65, -70 dBm
- 대표 임계값: -65 dBm
- 경계 영역: |GT RSS - 임계값| <= 10 dB를 만족하는 픽셀

## 저장소 구조

```text
ikeee/                        데이터셋, 모델, 학습 코드 (Python 패키지)
  bratlab_dataset.py          BRAT-Lab 데이터셋, 고정 seed 기반 70/10/20 분할
  bratlab_train.py            학습, 단일 체크포인트 평가
  model/
    radiounet.py              RadioUNet-S / M
    transpathnet_outdoor.py   TransPathNet-S / B (실외용, 2채널 입력)
    transnext_backbone.py     TransPathNet용 경량 TransNeXt backbone
    emcad_decoder.py          TransPathNet용 EMCAD decoder

scripts/
  run_multi_seed_holdout_experiments.sh           seed별 학습, 평가, 집계
  resume_seed43_44_remaining.sh                   seed 43/44 실험 재개
  aggregate_seed_metrics.py                       seed 간 평균, 표준편차 계산
  final_eval_metrics_fixed_threshold_heatmap.py   고정 임계값 지표, heatmap, 표
  boundary_only_cdf_analysis.py                   경계 영역 오차, miss rate CDF
  boundary_only_cdf_analysis_x10.py               위와 동일, 오차 축 0-10 dB
  final_qualitative_split_allpairs_v11.py         오차 맵, 음영 검출 그림 (단일 주파수)
  qualitative_shared_sample_1750_4750_5750.py     오차 맵, 음영 검출 그림 (공통 샘플, 3개 주파수)
  run_1750_4750_5750_shared_pipeline.sh           대표 결과 생성 파이프라인
  final_samplewise_cdf_v3.py                      샘플별 miss rate CDF (CSV 입력)
  paper_missrate_cdf_builder.py                   miss rate CDF 그림, 요약 표 (CSV 입력)
  plot_existing_cdf_font_plus10.py                CDF 그림 글꼴 확대 (CSV 입력)
  plot_fixed_threshold_heatmap_font_plus10.py     heatmap 글꼴 확대 (CSV 입력)
  plot_rmse_vs_frequency_paper.py                 주파수별 RMSE 그림 (CSV 입력)

results/
  fixed_threshold_metrics/
  samplewise_cdf/
  boundary_cdf/
  qualitative_3750/
  representative_1750_4750_5750/
```

## 데이터와 체크포인트

- 데이터셋과 체크포인트는 저장소에 포함하지 않는다.
- 기본 경로는 저장소 루트의 `data/`, `model_checkpoints/`이다.

```text
data/
  stations_position.txt                       샘플별 송신기 위치 (row1 col1 row2 col2 row3 col3)
  buildings_position/buildings_{n}.mat
  receivedpower_{freq}MHz_mat/TOTAL_REC_{freq}MHz_{n}.mat
model_checkpoints/
  checkpoints_{model}_{size}_{freq}/best.pth   예) checkpoints_transpathnet_base_4750/best.pth
```

- n = 1, ..., 2000
- NaN 값이 있거나 50 x 50 픽셀보다 작은 샘플은 제외한다 (4750 MHz에서 12개).
- 테스트 샘플 수: 400개 (4750 MHz는 399개)

## 학습

- 저장소 루트에서 실행한다.

```bash
python -m ikeee.bratlab_train --mode train \
    --root data --freq 4750 --model radiounet --size small \
    --batch_size 8 --epochs 100 --lr 1e-4 \
    --save_dir model_checkpoints/checkpoints_radiounet_small_4750
```

- 모델 옵션: `--model radiounet --size small|medium`, `--model transpathnet --size small|base`
- 최적화: AdamW (weight decay 1e-4), cosine annealing (eta_min 1e-6), gradient clipping 1.0
- 손실 함수: fine 출력에 0.6 * MSE + 0.3 * (1 - SSIM) + 0.1 * MSE(aux)를 적용하고, coarse 출력에 같은 손실의 0.4배를 더한다.
- `best.pth`는 validation loss가 가장 낮은 체크포인트이다.
- seed 42 체크포인트(`results/`에 사용)의 학습 설정

| 모델 | Batch size | Learning rate | Epoch |
|---|---|---|---|
| RadioUNet-S | 8 | 1e-4 | 100 |
| RadioUNet-M | 4 | 1e-4 | 100 |
| TransPathNet-S | 2 | 1e-4 | 100 |
| TransPathNet-B | 2 | 1e-4 | 100 |

- 4750 MHz RadioUNet-S는 `checkpoints_radiounet_small_4750_v2/best.pth`(lr 3e-5)가 있으면 평가 스크립트에서 이 체크포인트를 사용한다.

### 단일 체크포인트 평가

- 건물을 제외한 픽셀에서 RMSE, MAE, bias(dB)를 계산한다.

```bash
python -m ikeee.bratlab_train --mode eval \
    --root data --freq 4750 --model radiounet --size small \
    --ckpt model_checkpoints/checkpoints_radiounet_small_4750/best.pth \
    --result_dir runs/eval_radiounet_small_4750
```

- `--result_dir`의 기본값은 `./results`이다. `results/`를 그대로 두려면 다른 경로를 지정한다.

### 분할 seed를 바꾼 반복 holdout 실험

```bash
DRY_RUN=1 bash scripts/run_multi_seed_holdout_experiments.sh   # 실행할 명령만 출력
bash scripts/run_multi_seed_holdout_experiments.sh             # seed 42-46, 1750/4750/5750 MHz
```

- 환경 변수(`SEEDS`, `FREQS`, `MODELS`, `ROOT`, `OUT_DIR`, `EPOCHS` 등)로 설정을 변경한다.
- 결과는 `results_multi_seed_holdout/`에 저장한다.
- seed 43, 44에서 TransPathNet은 `TRANSPATHNET_BATCH_SIZE=2 TRANSPATHNET_ACCUM_STEPS=4 TRANSPATHNET_AMP=1`로, RadioUNet은 AMP 없이 batch size 8로 학습했다.

## 대표 결과 재현

```bash
bash scripts/run_1750_4750_5750_shared_pipeline.sh

# 데이터와 체크포인트가 다른 경로에 있는 경우
ROOT=/path/to/data CKPT_DIR=/path/to/model_checkpoints \
    bash scripts/run_1750_4750_5750_shared_pipeline.sh
```

- 결과는 `results_1750_4750_5750_regenerated/`에 저장한다 (`OUT_DIR`로 변경 가능).

## 결과 파일별 생성 스크립트

| 결과 | 스크립트 |
|---|---|
| `results/fixed_threshold_metrics/heatmap_fixed_threshold_th-65.png`, `*_metrics_fixed_threshold.csv` | `final_eval_metrics_fixed_threshold_heatmap.py` |
| `results/fixed_threshold_th-65.png`, `results/fixed_threshold_metrics/heatmap_fixed_threshold_th-65_font_plus10.png` | `plot_fixed_threshold_heatmap_font_plus10.py` |
| `results/samplewise_cdf/fig_cdf_missrate_*.png` | `final_samplewise_cdf_v3.py` |
| `results/samplewise_cdf/samplewise_miss_rate_rows.csv`, `cdf_summary_table.csv` | `paper_missrate_cdf_builder.py` |
| `results/samplewise_cdf/samplewise_error_rates.csv` | 이전 버전 스크립트 (참고 사항 참조) |
| `results/boundary_cdf/fig_boundary_{error,combined}_cdf_th-65_bw10.png` | `boundary_only_cdf_analysis.py` |
| `results/boundary_cdf/`의 나머지 파일 | `boundary_only_cdf_analysis_x10.py` |
| `results/qualitative_3750/` | `final_qualitative_split_allpairs_v11.py` (3750 MHz, sample 1486) |
| `results/representative_1750_4750_5750/{metrics,miss_cdf,boundary_cdf,qualitative}/` | `run_1750_4750_5750_shared_pipeline.sh` (`final_eval_metrics_fixed_threshold_heatmap.py`, `final_samplewise_cdf_v3.py`, `boundary_only_cdf_analysis_x10.py`, `qualitative_shared_sample_1750_4750_5750.py`) |
| `results/representative_1750_4750_5750/*_font_plus10/` | `plot_existing_cdf_font_plus10.py` |

- 체크포인트를 사용하는 결과는 모두 같은 테스트 샘플(seed 42 분할)에서 계산했다.

## 참고 사항

- `model_checkpoints/`와 원본 데이터셋 파일은 저장소에서 제외한다.
- `results/`의 CSV 파일은 재현과 시각화를 위해 선별한 결과이다.
- 그림은 논문과 연구실 미팅에 사용한 대표 결과이다.
- 일부 그림은 이전 버전의 plotting 스크립트로 생성했다. 현재 스크립트로 다시 생성하면 글꼴 크기나 레이아웃이 다를 수 있다.
- `results/samplewise_cdf/samplewise_error_rates.csv`는 주파수별 임계값(1750: -61.7, 2750: -64.9, 3750: -67.4, 4750: -69.5, 5750: -71.1 dBm)을 사용한 이전 스크립트의 결과이다. 해당 스크립트는 남아 있지 않고, 고정 임계값 버전은 `final_samplewise_cdf_v3.py`이다.
- `plot_rmse_vs_frequency_paper.py`는 기본적으로 5개 주파수 평가 결과인 `summary_metrics_long.csv`를 읽으며, 이 파일은 저장소에 없다. 실행할 때 `--summary_csv`와 `--out`을 지정한다.
- `--summary_csv`에는 `rmse_dbm_mean`, `rmse_dbm_std` 열이 있는 summary CSV를 사용한다. 예) `results/fixed_threshold_metrics/summary_metrics_fixed_threshold.csv`

## 참고 자료

- 모델 코드는 다음 연구를 바탕으로 구현했다.
- RadioUNet: R. Levie et al., "RadioUNet: Fast Radio Map Estimation With Convolutional Neural Networks," IEEE Transactions on Wireless Communications, 2021 (arXiv:1911.09002)
- TransPathNet: https://github.com/LiXin97/TransPathNet
- TransNeXt: https://github.com/DaiShiResearch/TransNeXt (Apache-2.0)
- EMCAD: Rahman et al., CVPR 2024, https://github.com/SLDGroup/EMCAD
