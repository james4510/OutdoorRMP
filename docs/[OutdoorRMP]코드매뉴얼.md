# Radio Map 예측 기반 음영 영역 검출 코드 매뉴얼

5G 특화망의 미래 군통신망 적용 연구 및 시험평가 기술지원

| 항목 | 내용 |
|---|---|
| 연구용역명 | 5G 특화망의 미래 군통신망 적용 연구 및 시험평가 기술지원 |
| 주관기관 | 국방과학연구소 |
| 수행기관 | LIG D&A |
| 위탁연구기관 | 가천대학교 INC 연구실 |
| 문서 버전 | 1.0 |
| 작성일 | 2026-10-16 |

## 목차

- 1. 개요
  - 1.1 문서 목적과 과제 정보
  - 1.2 문제 정의와 모델
  - 1.3 대표 결과와 코드 대응
  - 1.4 모델링 범위
  - 1.5 라이선스와 참조 구현
- 2. 코드 구성
  - 2.1 구성요소 구분
  - 2.2 디렉터리와 역할
  - 2.3 실행 환경과 외부 의존성
  - 2.4 주요 클래스·함수
- 3. 데이터셋과 입력
  - 3.1 데이터 파일 구성
  - 3.2 입력 채널과 정답
  - 3.3 전처리와 증강
  - 3.4 유효 샘플과 분할
- 4. 모델
  - 4.1 공통 구조
  - 4.2 RadioUNet
  - 4.3 TransPathNet
  - 4.4 모델 선택과 체크포인트
- 5. 학습과 평가(bratlab_train.py)
  - 5.1 실행 흐름
  - 5.2 손실 함수
  - 5.3 CLI 옵션
  - 5.4 학습 출력 파일
  - 5.5 평가 모드
  - 5.6 난수와 재현성
- 6. 분석 스크립트와 출력 지표
  - 6.1 파이프라인 구성
  - 6.2 공통 규칙
  - 6.3 고정 임계값 지표(final_eval_metrics_fixed_threshold_heatmap.py)
  - 6.4 샘플별 miss rate CDF(final_samplewise_cdf_v3.py)
  - 6.5 경계 영역 CDF(boundary_only_cdf_analysis_x10.py)
  - 6.6 정성 그림(qualitative_shared_sample_1750_4750_5750.py)
  - 6.7 반복 holdout 실험
  - 6.8 보조 스크립트
  - 6.9 결과 해석 예
- 7. 코드 수정과 확장
  - 7.1 수정 위치
  - 7.2 새 모델 추가 순서
  - 7.3 실습 예제: 임계값 -75 dBm 추가
  - 7.4 변경 사항 점검
- 8. 제약사항과 유의사항
  - 8.1 모델 범위
  - 8.2 결과 해석 유의
  - 8.3 재현성
  - 8.4 실행 도구
  - 8.5 지원 범위
- 부록 A. CLI 옵션 전체
  - A.1 check_setup.py
  - A.2 aggregate_seed_metrics.py
  - A.3 run_multi_seed_holdout_experiments.sh 환경 변수
  - A.4 final_qualitative_split_allpairs_v11.py
  - A.5 paper_missrate_cdf_builder.py
  - A.6 그림 재작도 스크립트
- 부록 B. 결과 CSV 컬럼
  - B.1 per_sample_metrics_fixed_threshold.csv(6.3)
  - B.2 summary_metrics_fixed_threshold.csv(6.3)
  - B.3 per_sample_boundary_metrics.csv, boundary_cdf_summary.csv(6.5)
  - B.4 miss rate CDF 파일(6.4)
  - B.5 정성 그림 파일(6.6)
  - B.6 반복 실험 집계 파일(6.7)

## 1. 개요

### 1.1 문서 목적과 과제 정보

본 문서는 국방과학연구소가 주관하고 LIG D&A가 수행하는 『5G 특화망의 미래 군통신망 적용 연구 및 시험평가 기술지원』 과제에서 가천대학교 INC 연구실이 위탁 수행한 Radio Map 예측 기반 음영 영역 검출(Radio Map Prediction-based Shadow Region Detection in Sub-7 GHz Environments) 소프트웨어의 코드 매뉴얼이다.

이 문서는 다음 내용을 설명한다.

- 데이터셋 로더, 모델 4종, 학습·평가 코드, 분석 스크립트의 구조와 실행 흐름
- 입력 옵션과 출력 파일·지표
- 코드를 수정·확장하는 방법과 결과 해석 시 유의사항

설치와 실행 절차는 『설치 매뉴얼』에 있다. 본 문서에서 쓰는 용어는 다음과 같다.

- 라디오맵: 지도 위 각 픽셀의 수신 신호 세기(RSS, dBm)를 나타낸 2차원 맵. 모델의 정답(GT)이자 예측 대상
- 음영 영역: RSS가 임계값보다 낮은 픽셀(건물 제외). 고정 임계값은 -70, -65, -60 dBm
- 경계 영역: 정답 RSS가 임계값에서 ±10 dB 안에 있는 픽셀
- 분할 seed: 학습·검증·테스트 분할에 쓰는 난수 seed. 제공 체크포인트와 결과는 모두 seed 42
- 환경 이미지: 패키지에 포함된 Docker 이미지 `outdoor-rmp:latest`

| 항목 | 내용 |
|---|---|
| 대상 패키지 | `LIG_RMP_SHADOW_v1.0` |
| 대상 소스 | `ikeee/`(데이터셋 로더 1개, 학습·평가 1개, 모델 4개 모듈), `scripts/`(설치 확인 1개, 파이프라인 셸 1개, 실험 셸 2개, 분석·그림·집계 파이썬 11개) |
| 기반 소프트웨어 | PyTorch 2.2.0 + CUDA 11.8, torchvision 0.17.0, numpy 1.26, scipy 1.15.3, matplotlib 3.10.0, pillow, pytorch-msssim 1.0.0 |

표기 원칙

- 코드 위치는 파일명과 클래스·함수·옵션 이름으로 표시한다. `ikeee/bratlab_train.py`의 함수는 `bratlab_train.train`처럼 적는다.
- 모델은 표시 이름(RadioUNet-S)과 태그(`radiounet-small`)를 함께 쓴다. 태그는 스크립트 옵션과 결과 CSV의 `model_tag` 열에 쓰인다.
- 일반적인 PyTorch 설명은 생략하고, 이 패키지의 코드가 구현한 내용만 설명한다.

### 1.2 문제 정의와 모델

건물 배치와 송신기 3개의 위치가 주어졌을 때 256 × 256 라디오맵(RSS, -90 ~ 0 dBm)을 예측하고, 예측 라디오맵에 고정 임계값을 적용해 음영 영역을 검출한다. 정답 라디오맵에 같은 임계값을 적용한 음영 영역과 비교해 검출 성능을 평가한다. 모델은 다음 네 가지이다.

| 모델 | 태그 | 파라미터 | 구조 | 본 문서 |
|---|---|---|---|---|
| RadioUNet-S | `radiounet-small` | 34.92M | CNN. U-Net 2개(coarse, fine), base 채널 48, 깊이 4 | 4.2 |
| RadioUNet-M | `radiounet-medium` | 96.98M | CNN. 같은 구조, base 채널 80 | 4.2 |
| TransPathNet-S | `transpathnet-small` | 38.09M | Transformer. TransNeXt 백본(임베딩 64/128/256/512, 깊이 2/2/6/2) + EMCAD 디코더, 2단계 | 4.3 |
| TransPathNet-B | `transpathnet-base` | 111.44M | Transformer. 임베딩 96/192/384/768, 깊이 3/3/9/3 | 4.3 |

공통 구성은 다음과 같다.

- 입력 2채널: 건물 마스크(1 = 건물), 송신기 3개 위치에 Gaussian 필터(σ = 4)를 적용한 맵
- 출력: [-90, 0] dBm을 [0, 1]로 정규화한 라디오맵. coarse 예측 후 coarse 결과를 입력에 붙여 fine 예측을 수행하는 2단계 구조
- 주파수: 1750, 2750, 3750, 4750, 5750 MHz. 주파수마다 모델을 따로 학습한다
- 손실: 0.6·MSE + 0.3·(1 − SSIM) + 0.1·MSE(aux)를 fine 출력에, 같은 손실의 0.4배를 coarse 출력에 적용(5.2)

### 1.3 대표 결과와 코드 대응

패키지의 `results/`에 들어 있는 논문 결과와 이를 만든 스크립트의 대응이다. 모든 결과는 seed 42의 테스트 분할(주파수당 400 샘플, 4750 MHz는 399 샘플)로 만들었다.

| # | 결과 폴더·파일 | 내용 | 스크립트 | 본 문서 |
|---|---|---|---|---|
| 1 | `representative_1750_4750_5750/metrics/` | 고정 임계값(-70, -65, -60 dBm) 지표 CSV, heatmap, 표 | `final_eval_metrics_fixed_threshold_heatmap.py` | 6.3 |
| 2 | `representative_1750_4750_5750/miss_cdf/` | 샘플별 miss rate CDF(주 그림, 1750 MHz 확대) | `final_samplewise_cdf_v3.py` | 6.4 |
| 3 | `representative_1750_4750_5750/boundary_cdf/` | 경계 영역 RSS 오차·miss rate CDF | `boundary_only_cdf_analysis_x10.py` | 6.5 |
| 4 | `representative_1750_4750_5750/qualitative/` | 세 주파수 공통 샘플(770)의 오차 맵과 음영 검출 결과 | `qualitative_shared_sample_1750_4750_5750.py` | 6.6 |
| 5 | `representative_1750_4750_5750/*_font_plus10/` | 2, 3의 그림을 큰 글꼴로 다시 그린 것 | `plot_existing_cdf_font_plus10.py` | 6.8 |
| 6 | `fixed_threshold_metrics/`, `fixed_threshold_th-65.png` | -65 dBm heatmap(논문 그림용, 큰 글꼴) | 1과 `plot_fixed_threshold_heatmap_font_plus10.py` | 6.8 |
| 7 | `samplewise_cdf/` | 5개 주파수 miss rate CDF와 요약 표 | `final_samplewise_cdf_v3.py`, `paper_missrate_cdf_builder.py` | 6.8 |
| 8 | `boundary_cdf/` | -65 dBm 경계 영역 CDF(x축 12 dB 판과 10 dB 판) | `boundary_only_cdf_analysis.py`, `_x10.py` | 6.5 |
| 9 | `qualitative_3750/` | 3750 MHz 샘플 1486의 오차 맵과 음영 검출(GT 기준 TP/FN) | `final_qualitative_split_allpairs_v11.py` | 6.8 |

- 1 ~ 4는 `scripts/run_1750_4750_5750_shared_pipeline.sh`가 한 번에 만든다(『설치 매뉴얼』 5.2). 5 ~ 9는 논문 그림 작성용 보조 결과로 각 스크립트를 직접 실행한다.
- 결과 1의 CSV는 이전 판 스크립트로 만들어 `seed` 열이 없다. 현재 스크립트는 `seed` 열을 첫 열로 쓴다(8.2).

### 1.4 모델링 범위

- 2차원 라디오맵만 다룬다. 입력은 건물 유무와 송신기 위치뿐이며 건물 높이·재질, 안테나 패턴, 지형은 입력에 없다.
- 송신기는 샘플마다 3개로 고정되어 있다(`stations_position.txt`의 6개 열). 개수를 바꾸려면 데이터셋 로더를 수정한다(7.1).
- 모델 입출력 크기는 256 × 256으로 고정하며, 원본 라디오맵은 bilinear 보간으로 축소·확대한다. 지표는 256 × 256 해상도에서 계산한다.
- 음영 판정은 픽셀 단위 임계값 비교이며, 건물 픽셀은 모든 지표에서 제외한다.
- 주파수 간 전이 학습이나 다중 주파수 모델은 범위 밖이다. 주파수마다 별도 체크포인트를 쓴다.

### 1.5 라이선스와 참조 구현

| 구성요소 | 출처 | 본 패키지에서의 취급 |
|---|---|---|
| RadioUNet | Levie et al., "RadioUNet", IEEE TWC 2021 (arXiv:1911.09002) | 논문 구조를 참고해 2채널 입력·2단계 U-Net으로 재구현(`ikeee/model/radiounet.py`) |
| TransPathNet | LiXin97/TransPathNet (ICASSP 2025 실내 라디오맵 챌린지) | 실내 전용 입력(반사·투과·거리 등 7채널)을 제거하고 2채널 입력으로 재구성(`transpathnet_outdoor.py`) |
| TransNeXt | DaiShiResearch/TransNeXt (Apache 2.0) | Aggregated Attention 구조를 경량화해 백본으로 재구현. 사전 학습 가중치 없이 처음부터 학습(`transnext_backbone.py`) |
| EMCAD | Rahman et al., CVPR 2024, SLDGroup/EMCAD (BSD 3-Clause) | 다중 스케일 디코더를 재구현(`emcad_decoder.py`) |
| PyTorch, torchvision, numpy, scipy, matplotlib, pillow, pytorch-msssim | 각 프로젝트의 오픈소스 라이선스(BSD, PSF, MIT 등) | 환경 이미지에 설치된 패키지로 사용 |

## 2. 코드 구성

### 2.1 구성요소 구분

| 구분 | 범위 | 파일 수 | 기준 |
|---|---|---|---|
| 신규 코드(핵심) | `ikeee/bratlab_dataset.py`, `ikeee/bratlab_train.py` | 2 | 데이터셋 로더, 학습·평가. 본 과제에서 작성 |
| 신규 코드(모델) | `ikeee/model/*.py` | 4 | 참조 구현의 구조를 2채널 입력에 맞게 재구현(1.5) |
| 신규 코드(분석) | `scripts/*.py`, `scripts/*.sh` | 15 | 지표 계산, CDF, 정성 그림, 반복 실험, 설치 확인 |
| 실행 환경 | `Dockerfile`, `docker-compose.yml`, `docker/` | 4 | 환경 이미지 정의와 실행 설정(『설치 매뉴얼』 3.4) |
| 외부 패키지 | PyTorch 등 | – | 환경 이미지에 설치. 소스를 포함하지 않음 |

### 2.2 디렉터리와 역할

```text
LIG_RMP_SHADOW_v1.0/
├── ikeee/
│   ├── bratlab_dataset.py          데이터셋 로더: BRATLabPaths, BRATLabDataset, get_dataloaders(3장)
│   ├── bratlab_train.py            학습·평가 진입점: train, evaluate, RadioLoss, build_model(5장)
│   └── model/
│       ├── radiounet.py            RadioUNet(UNetModule 2개), build_radiounet(4.2)
│       ├── transpathnet_outdoor.py TransPathNetOutdoor(TransPathUnit 2개), build_transpathnet(4.3)
│       ├── transnext_backbone.py   TransNeXtBackbone(4단계 Aggregated Attention)(4.3)
│       └── emcad_decoder.py        EMCADDecoder(MSCAM, LGAG, 보조 출력)(4.3)
├── scripts/
│   ├── check_setup.py              설치 확인: 패키지·GPU·코드·데이터셋·체크포인트 점검
│   ├── run_1750_4750_5750_shared_pipeline.sh   대표 결과 파이프라인(아래 4개를 순서대로 실행)(6.1)
│   ├── final_eval_metrics_fixed_threshold_heatmap.py   고정 임계값 지표·heatmap(6.3)
│   ├── final_samplewise_cdf_v3.py                      샘플별 miss rate CDF(6.4)
│   ├── boundary_only_cdf_analysis_x10.py               경계 영역 오차·miss rate CDF(6.5)
│   ├── qualitative_shared_sample_1750_4750_5750.py     공통 샘플 정성 그림(6.6)
│   ├── run_multi_seed_holdout_experiments.sh           분할 seed 반복 실험(6.7)
│   ├── resume_seed43_44_remaining.sh                   seed 43, 44 이어서 실행(6.7)
│   ├── aggregate_seed_metrics.py                       seed별 지표 집계(6.7)
│   ├── boundary_only_cdf_analysis.py, final_qualitative_split_allpairs_v11.py,
│   │   paper_missrate_cdf_builder.py, plot_existing_cdf_font_plus10.py,
│   │   plot_fixed_threshold_heatmap_font_plus10.py, plot_rmse_vs_frequency_paper.py   논문 그림 보조(6.8)
├── results/                        기준 결과(1.3)
├── data/                           데이터셋(3.1). stations_position.txt, buildings_position/, receivedpower_<주파수>MHz_mat/
└── model_checkpoints/              체크포인트. checkpoints_<모델>_<크기>_<주파수>/best.pth
```

- `ikeee`는 설치하지 않는 일반 패키지이다. 패키지 루트에서 `python -m ikeee.bratlab_train`으로 실행하며, 환경 이미지는 `PYTHONPATH=/workspace`로 이를 보장한다. `scripts/`의 파이썬 파일은 `ikeee/bratlab_dataset.py`가 있는 상위 폴더를 찾아 `sys.path`에 스스로 추가한다.
- `results/`는 읽기 전용으로 쓴다. 재현 결과는 `results_1750_4750_5750_regenerated/` 등 다른 폴더에 생긴다.

### 2.3 실행 환경과 외부 의존성

| 항목 | 내용 |
|---|---|
| 베이스 이미지 | `pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime`(Python 3.10, torch 2.2.0, torchvision 0.17.0, numpy, pillow 포함) |
| 추가 패키지(`docker/requirements.txt`) | `numpy<2`, `scipy==1.15.3`, `matplotlib==3.10.0`, `pytorch-msssim==1.0.0` |
| 환경 변수(`Dockerfile`) | `PYTHONPATH=/workspace`, `MPLBACKEND=Agg`, `MPLCONFIGDIR=/tmp/matplotlib`, `HOME=/tmp`, `PYTHONUNBUFFERED=1` |
| 선택 의존성 | `pytorch_msssim`이 없으면 `bratlab_train`이 경고를 내고 SSIM 항 없이 동작한다(손실과 지표가 달라지므로 반드시 설치) |
| GPU | CUDA가 없으면 자동으로 CPU에서 실행한다(`torch.cuda.is_available()`). 학습은 GPU 전제 |

### 2.4 주요 클래스·함수

| 파일 | 식별자 | 역할 |
|---|---|---|
| `ikeee/bratlab_dataset.py` | `BRATLabPaths` | 데이터 파일 경로 규칙(`rm_path(n)`, `bp_path(n)`)(3.1) |
| 〃 | `load_stations`, `make_building_mask`, `make_tx_map`, `make_gt` | 송신기 위치 읽기, 건물 마스크, 송신기 맵, 정답 정규화(3.2) |
| 〃 | `BRATLabDataset` | 샘플 로드와 유효성 검사, 입력 2채널·정답 생성, 크기 조정, 증강(3.2, 3.3) |
| 〃 | `get_dataloaders` | 고정 seed 분할(7:1:2)과 DataLoader 3개 생성(3.4) |
| `ikeee/bratlab_train.py` | `set_seed` | Python·NumPy·PyTorch seed 고정, 선택적 deterministic cuDNN(5.6) |
| 〃 | `RadioLoss` | MSE + (1 − SSIM) + aux MSE 가중 손실(5.2) |
| 〃 | `build_model`, `model_tag` | `--model`, `--size` → 모델 객체·태그(4.4) |
| 〃 | `train` | 학습 루프: AMP, 누적, 기울기 클리핑, 검증, `best.pth` 저장, 곡선 그림(5.1) |
| 〃 | `evaluate` | 테스트 분할 평가: dBm 지표, SSIM, 예측 그림(5.5) |
| 〃 | `compute_metrics`, `compute_metrics_dbm`, `denorm` | 정규화 단위 지표(검증용), dBm 지표(평가용), 역정규화 |
| `ikeee/model/radiounet.py` | `UNetModule`, `RadioUNet`, `build_radiounet` | 단일 U-Net, 2단계 결합, 크기별 생성(4.2) |
| `ikeee/model/transpathnet_outdoor.py` | `TransPathUnit`, `TransPathNetOutdoor`, `build_transpathnet` | 백본 + 디코더 단위, 2단계 결합, 크기별 생성(4.3) |
| `ikeee/model/transnext_backbone.py` | `AggregatedAttention`, `TransNeXtBlock`, `PatchEmbed`, `TransNeXtBackbone` | 어텐션, 블록, 패치 임베딩, 4단계 백본(4.3) |
| `ikeee/model/emcad_decoder.py` | `MSCAM`, `LGAG`, `EMCADBlock`, `EMCADDecoder` | 다중 스케일 어텐션, 게이트, 디코더 블록, 디코더(4.3) |
| `scripts/final_eval_metrics_fixed_threshold_heatmap.py` | `resolve_ckpt_path`, `load_model`, `compute_shadow_metrics`, `evaluate`, `summarize` | 체크포인트 선택(v2 규칙), 로드, 픽셀 지표, 샘플 루프, 요약·micro 지표(6.3) |
| `scripts/final_samplewise_cdf_v3.py` | `cdf_fraction`, `choose_zoom_window` | CDF 계산, 확대 구간 자동 선택(6.4) |
| `scripts/boundary_only_cdf_analysis_x10.py` | `compute_boundary_metrics`, `empirical_cdf` | 경계 영역 지표, 경험적 CDF(6.5) |
| `scripts/qualitative_shared_sample_1750_4750_5750.py` | `choose_shared_sample`, `find_sample` | 공통 샘플 선택 기준, 테스트 로더에서 샘플 찾기(6.6) |
| `scripts/aggregate_seed_metrics.py` | `main` | seed별 요약 CSV 집계(6.7) |
| `scripts/check_setup.py` | `main` | 5단계 점검과 `Result: OK/FAIL` 판정 |

## 3. 데이터셋과 입력

### 3.1 데이터 파일 구성

```text
data/
├── stations_position.txt                 샘플별 송신기 3개 위치. 한 줄에 6개 정수 [row1 col1 row2 col2 row3 col3], n번째 줄 = 샘플 n
├── buildings_position/
│   └── buildings_<n>.mat                 건물 배치(0 ~ 255 스케일, 127 초과 = 건물)
└── receivedpower_<주파수>MHz_mat/
    └── TOTAL_REC_<주파수>MHz_<n>.mat     수신 전력 라디오맵(dBm). 주파수 1750, 2750, 3750, 4750, 5750
```

- `n`은 1 ~ 2000이다. `BRATLabPaths(root, freq)`가 위 경로 규칙을 만들고, `.mat` 파일에서는 이름이 `__`로 시작하지 않는 첫 변수를 읽는다(변수 이름을 가정하지 않음).
- 라디오맵과 건물 배치의 원본 크기는 샘플마다 다를 수 있으며, 건물 배치가 라디오맵과 크기가 다르면 nearest 보간으로 라디오맵 크기에 맞춘다.
- 라디오맵 값이 정확히 0인 픽셀은 정규화 후에도 0(= -90 dBm)으로 남는다. 별도로 제외하지 않는다.

### 3.2 입력 채널과 정답

| 텐서 | 크기 | 생성 | 의미 |
|---|---|---|---|
| `x[0]` 건물 마스크 | (256, 256) | `make_building_mask`: 값 > 127 → 1.0 | 1 = 건물, 0 = 그 외. 지표 계산에서 `< 0.5`인 픽셀만 유효 픽셀 |
| `x[1]` 송신기 맵 | (256, 256) | `make_tx_map`: 송신기 3개 위치에 1을 놓고 `gaussian_filter(sigma=4.0)` 후 최댓값으로 나눔 | 최댓값 1의 연속 맵. 송신기 위치가 맵 밖이면 가장자리로 클리핑 |
| `y` 정답 | (1, 256, 256) | `make_gt`: `(RSS − (−90)) / (0 − (−90))`을 [0, 1]로 클리핑 | 정규화 라디오맵. 역정규화 `denorm(t) = t × 90 − 90` |

- `BRATLabDataset.__getitem__`은 `(x, y, n)`을 돌려준다. `n`은 1부터 시작하는 원본 샘플 번호이며 결과 CSV의 `sample_id`와 그림 파일명(`sample770`)에 쓰인다.
- 정규화 범위 -90 ~ 0 dBm은 `BRATLabDataset(min_dbm, max_dbm)`과 분석 스크립트의 `--min_dbm`, `--max_dbm` 옵션으로 바꿀 수 있다. 학습과 평가에서 같은 값을 써야 한다.

### 3.3 전처리와 증강

- 크기 조정: 입력과 정답 모두 `torchvision.transforms.functional.resize`로 256 × 256(`img_size`)에 맞춘다(bilinear, antialias). 건물 마스크는 보간으로 경계가 0 ~ 1 사이 값이 되므로 이후 코드는 `< 0.5`, `> 0.5`로 이진화한다.
- 증강(학습 분할만): 50 % 확률 좌우 반전, 50 % 확률 상하 반전, 0 ~ 3회 90° 회전(`torch.rot90`). 입력과 정답에 같은 변환을 적용한다. 검증·테스트 분할은 증강하지 않는다.
- 증강은 Python `random` 모듈을 쓰며, DataLoader worker마다 `seed + worker_id`로 초기화한다(`seed_worker`).

### 3.4 유효 샘플과 분할

- 유효성 검사(`BRATLabDataset.__init__`): 라디오맵에 NaN이 있거나 높이·너비가 50 픽셀 미만이거나 파일 로드에 실패한 샘플은 제외한다. 4750 MHz에서 12개가 제외되어 1,988 샘플, 나머지 주파수는 2,000 샘플이다. 파일이 없는 번호는 조용히 건너뛴다.
- 분할(`get_dataloaders`): 유효 샘플 수 n에 대해 학습 `int(0.7n)`, 검증 `int(0.1n)`, 테스트 나머지. `torch.Generator().manual_seed(seed)`의 `randperm`으로 섞은 뒤 앞에서부터 나눈다. seed 42일 때 1400/200/400(4750 MHz는 1391/198/399)이다.
- 같은 seed이면 분할이 항상 같다. 제공 체크포인트와 `results/`는 모두 seed 42 분할이며, 분석 스크립트도 기본값 42를 쓴다. 분할을 바꾸려면 `--seed`(학습, 지표 스크립트)를 쓴다.
- DataLoader: 학습 로더는 `shuffle=True`에 `manual_seed(seed + 1000)` 생성기, 검증·테스트 로더는 순서 고정. 테스트 로더의 batch size는 인자와 무관하게 1로 고정된다. `pin_memory=True`.
- `get_dataloaders`는 기본 데이터셋과 분할별 데이터셋 3개를 따로 만들어 디렉터리를 4번 읽으므로 `[BRATLabDataset] ... 유효 샘플` 메시지가 4번 출력된다. 시작에 수 초 ~ 수십 초가 걸린다.

## 4. 모델

### 4.1 공통 구조

네 모델 모두 `forward(x)`가 `(coarse, fine, aux)`를 돌려준다. 각 출력은 (B, 1, 256, 256)이고 Sigmoid를 거쳐 [0, 1] 범위이다.

1. coarse 단계: 입력 `x`(2채널) → coarse 예측
2. fine 단계: `torch.cat([x, coarse.detach()], dim=1)`(3채널) → fine 예측. `detach()`로 fine 손실의 기울기가 coarse 단계로 흐르지 않게 한다
3. aux: 디코더 중간 해상도의 보조 예측(TransPathNet) 또는 fine과 같은 텐서(RadioUNet). 손실의 aux 항에만 쓰인다
- 평가와 분석 스크립트는 `fine`만 쓴다. `coarse`는 `evaluate`의 예측 그림에만 표시된다.
- 모델 선택은 `bratlab_train.build_model(model, size)`가 담당하며, 분석 스크립트는 같은 생성 함수(`build_radiounet`, `build_transpathnet`)를 직접 호출한다.

### 4.2 RadioUNet

`ikeee/model/radiounet.py`. U-Net 두 개를 직렬로 잇는 2단계 CNN이다.

| 식별자 | 역할 |
|---|---|
| `ConvBNReLU`, `DoubleConv` | 3×3 합성곱(bias 없음) + BatchNorm + ReLU. `DoubleConv`는 두 번 반복 |
| `UNetModule(in_ch, base_ch, depth)` | 인코더 `depth`단(채널 `base_ch × 2^i`, 단마다 `MaxPool2d(2)`), 병목(채널 2배), 디코더(`ConvTranspose2d` 2배 확대 + skip 연결 + `DoubleConv`), 출력 1×1 합성곱 + Sigmoid |
| `RadioUNet(in_ch=2, base_ch, depth=4)` | `coarse_unet = UNetModule(2, …)`, `fine_unet = UNetModule(3, …)`. 반환 `(coarse, fine, fine)` |
| `MODEL_CONFIGS`, `build_radiounet(size)` | `small: base_ch=48`, `medium: base_ch=80`(둘 다 `depth=4`) |

- small(base 48)의 채널 흐름: 인코더 48 → 96 → 192 → 384, 병목 768, 디코더 384 → 192 → 96 → 48, 출력 1. medium은 80부터 시작한다.
- 파라미터 수: small 34,922,354, medium 96,982,802(`count_params`로 측정).

### 4.3 TransPathNet

`ikeee/model/transpathnet_outdoor.py`. TransNeXt 백본과 EMCAD 디코더로 이루어진 단위(`TransPathUnit`) 두 개를 직렬로 잇는 2단계 Transformer 모델이다.

#### 4.3.1 백본(transnext_backbone.py)

| 식별자 | 역할 |
|---|---|
| `PatchEmbed(in_ch, out_ch, stride)` | 3×3 합성곱 두 번으로 `stride`배 축소 후 토큰화 + LayerNorm. 1단계 stride 4, 2 ~ 4단계 stride 2 |
| `AggregatedAttention(ch, num_heads, window_size=3)` | 전역 분기(softmax 정규화 선형 어텐션 근사)와 지역 분기(3×3 depthwise 합성곱 + BN)를 `global + sigmoid(mix) × local`로 결합. `mix`는 학습 파라미터 |
| `TransNeXtBlock` | pre-LayerNorm 어텐션 + MLP(확장 4배, GELU) 잔차 블록. stochastic depth(`drop_path_rate` 0.1을 블록마다 선형 증가) |
| `TransNeXtBackbone(in_ch, embeds, depths, heads)` | 4단계. 단계 i의 출력 `c_i`는 (B, embeds[i], H/4·2^i, W/4·2^i). 반환 `(c1, c2, c3, c4)` |

#### 4.3.2 디코더(emcad_decoder.py)

| 식별자 | 역할 |
|---|---|
| `MSCAM(ch, kernels=(1,3,5,7))` | 커널 크기별 depthwise 합성곱 병렬 + 채널 어텐션(평균 풀링 → MLP → Sigmoid) + 1×1 융합, 잔차 합 |
| `LGAG(ch_g, ch_x)` | 어텐션 게이트. skip 특징 `x`에 `sigmoid(psi(relu(W_g g + W_x x)))`를 곱함 |
| `EMCADBlock(in_ch, skip_ch, out_ch)` | `ConvTranspose2d` 2배 확대 → `LGAG`로 skip 가중 → 결합 후 `MSCAM` → 1×1 합성곱 + BN + GELU |
| `EMCADDecoder(embeds)` | `d3`(c4 + c3), `d2`(d3 + c2), `d1`(d2 + c1) 세 블록 → `up_final`(ConvTranspose 2회, 4배 확대) → `head`(3×3 + 1×1 합성곱, Sigmoid). `aux_head`는 `d2` 출력(H/8)에 1×1 합성곱 + Sigmoid 후 8배 bilinear 확대. 반환 `(out, aux)` |

#### 4.3.3 결합과 크기

| 식별자 | 역할 |
|---|---|
| `TransPathUnit(in_ch, embeds, depths, heads)` | 백본 + 디코더. 반환 `(out, aux)` |
| `TransPathNetOutdoor` | `coarse_net = TransPathUnit(2, …)`, `fine_net = TransPathUnit(3, …)`. 반환 `(coarse, fine, aux_fine)`. coarse 단계의 aux는 쓰지 않음 |
| `MODEL_CONFIGS`, `build_transpathnet(size)` | `small: embeds (64,128,256,512), depths (2,2,6,2), heads (2,4,8,16)`; `base: embeds (96,192,384,768), depths (3,3,9,3), heads (3,6,12,24)` |

- 파라미터 수: small 38,094,722, base 111,443,710. 모듈 docstring의 ~38.10M, ~111.45M은 반올림 차이이며 README와 본 문서는 38.09M, 111.44M을 쓴다.
- `transnext_backbone.py`의 `TransNeXtBackbone.__init__` docstring은 입력 채널을 3/4로 적고 있으나 실제 사용은 coarse 2채널, fine 3채널이다.

### 4.4 모델 선택과 체크포인트

| `--model --size` | 태그(`model_tag`) | 생성 함수 | 파라미터 | 체크포인트 폴더 |
|---|---|---|---|---|
| `radiounet small` | `radiounet-small` | `build_radiounet("small")` | 34.92M | `checkpoints_radiounet_small_<주파수>` |
| `radiounet medium` | `radiounet-medium` | `build_radiounet("medium")` | 96.98M | `checkpoints_radiounet_medium_<주파수>` |
| `transpathnet small` | `transpathnet-small` | `build_transpathnet("small")` | 38.09M | `checkpoints_transpathnet_small_<주파수>` |
| `transpathnet base` | `transpathnet-base` | `build_transpathnet("base")` | 111.44M | `checkpoints_transpathnet_base_<주파수>` |

- `build_model`은 `radiounet`에 `small|medium`, `transpathnet`에 `small|base`만 허용한다(assert).
- 체크포인트 폴더 이름은 태그의 `-`를 `_`로 바꾼 것이다. 분석 스크립트의 `resolve_ckpt_path`가 이 규칙으로 `best.pth`를 찾으며, 4750 MHz RadioUNet-S는 `checkpoints_radiounet_small_4750_v2/best.pth`가 있으면 그것을 우선한다(lr 3e-5로 다시 학습한 체크포인트).
- `best.pth`는 `torch.save`한 dict이며 키는 `epoch`, `model_state`, `optimizer`, `val_loss`, `model`, `size`, `config`(학습 옵션 전체)이다. 분석 스크립트는 `model_state`(또는 `state_dict`, 순수 state dict)를 읽고 `module.` 접두어를 제거한다.

## 5. 학습과 평가(bratlab_train.py)

### 5.1 실행 흐름

`python -m ikeee.bratlab_train --mode train`은 다음 순서로 동작한다.

1. `set_seed(seed, deterministic)`: Python, NumPy, PyTorch(CPU·CUDA) seed 고정
2. `get_dataloaders(root, freq, img_size, batch_size, num_workers, seed)`: 분할과 로더 생성(3.4)
3. `build_model(model, size).to(device)`: 모델 생성. 학습 가능 파라미터 수 출력
4. `AdamW(lr, weight_decay=1e-4)`, `CosineAnnealingLR(T_max=epochs, eta_min=1e-6)`, `GradScaler(enabled=amp)`, `RadioLoss()`
5. epoch 루프: 학습 배치마다 `autocast(enabled=amp)`로 순전파 → 손실(5.2) → `loss / accum_steps` 역전파 → `accum_steps`마다 `unscale_` → `clip_grad_norm_(1.0)` → `optimizer.step()`
6. 검증: `fine` 출력으로 `RadioLoss`(coarse 항 없음)와 정규화 단위 RMSE·MAE·SSIM 계산. 검증 손실이 최저이면 `best.pth` 저장
7. `scheduler.step()`, epoch 로그 출력
8. 종료: `training_curve.png`, `training_complete.txt` 저장
- epoch 로그 `[001/1] Train:0.1020 Val:0.0399 | RMSE:0.0534 MAE:0.0404 SSIM:0.8739 | 64.1s`(RadioUNet-S, 5750 MHz, batch 4, AMP, RTX 2080 Ti의 실제 출력)의 RMSE·MAE는 정규화된 [0, 1] 단위이고 건물 픽셀을 포함한다. dBm 단위의 지표는 평가 모드(5.5)에서만 계산한다.
- `--accum_steps`가 1보다 크면 그만큼의 배치를 모아 한 번 갱신한다. 유효 batch size = `batch_size × accum_steps`. 마지막 배치에서는 남은 기울기로 갱신한다.
- `--amp`는 CUDA에서만 켜진다. CPU에서는 자동으로 꺼진다.

### 5.2 손실 함수

`RadioLoss(w_mse=0.6, w_ssim=0.3, w_aux=0.1)`의 `forward(pred, aux, target)`은 다음을 계산한다.

```text
loss = 0.6 * MSE(pred, target) + 0.3 * (1 - SSIM(pred, target, data_range=1.0)) + 0.1 * MSE(aux, target)
```

학습 한 스텝의 손실은 fine과 coarse 출력에 각각 적용해 더한다. 검증 손실은 fine 항만 쓴다.

```text
train_loss = RadioLoss(fine, aux, y) + 0.4 * RadioLoss(coarse, aux, y)
val_loss   = RadioLoss(fine, aux, y)        # best.pth 선택 기준
```

- coarse 항에도 같은 `aux`(fine 단계의 보조 출력)가 들어가므로 aux MSE의 유효 가중치는 0.1 + 0.04 = 0.14이다. RadioUNet은 aux가 fine과 같은 텐서이므로 fine MSE 가중치가 그만큼 커진다.
- SSIM은 `pytorch_msssim.ssim`(11×11 Gaussian 창, `size_average=True`)이다. 패키지가 없으면 SSIM 항이 빠지고 로그의 SSIM이 0으로 표시된다.
- 손실과 검증 지표는 정규화 단위([0, 1])에서 계산한다. 정규화 단위 RMSE에 90을 곱하면 dBm 단위 RMSE가 된다(건물 픽셀 포함 기준).

### 5.3 CLI 옵션

| 옵션 | 자료형 | 기본값 | 의미 |
|---|---|---|---|
| `--mode` | `train\|eval` | `train` | 학습 또는 평가 |
| `--root` | str(필수) | – | 데이터셋 루트(`data`) |
| `--freq` | `1750\|2750\|3750\|4750\|5750` | `4750` | 주파수(MHz). 데이터 폴더 선택 |
| `--model` | `transpathnet\|radiounet` | `transpathnet` | 모델 계열 |
| `--size` | `small\|medium\|base` | `small` | 모델 크기(`radiounet: small\|medium`, `transpathnet: small\|base`) |
| `--img_size` | int | `256` | 입출력 크기 |
| `--batch_size` | int | `8` | 학습·검증 batch size(테스트 로더는 1 고정) |
| `--accum_steps` | int | `1` | 기울기 누적 스텝 수 |
| `--amp` | flag | 꺼짐 | CUDA 자동 혼합 정밀도 사용 |
| `--epochs` | int | `100` | epoch 수(cosine 스케줄의 주기) |
| `--lr` | float | `1e-4` | 초기 학습률 |
| `--num_workers` | int | `4` | DataLoader worker 수 |
| `--save_dir` | str | `./checkpoints` | 학습 산출물 폴더 |
| `--seed` | int | `42` | 분할·학습 seed |
| `--deterministic` | flag | 꺼짐 | cuDNN 결정적 모드(`benchmark=False`, `deterministic=True`) |
| `--ckpt` | str | `./checkpoints/best.pth` | (eval) 체크포인트 경로 |
| `--result_dir` | str | `./results` | (eval) 예측 그림 저장 폴더. 기준 결과 보존을 위해 다른 경로 지정 |
| `--n_vis` | int | `10` | (eval) 그림을 저장할 샘플 수(테스트 로더 순서대로 앞에서부터) |

### 5.4 학습 출력 파일

| 파일 | 내용 |
|---|---|
| `best.pth` | 검증 손실 최저 epoch의 dict: `epoch`, `model_state`, `optimizer`, `val_loss`, `model`, `size`, `config`. 옵티마이저 상태를 포함하므로 가중치만의 2 ~ 3배 크기(RadioUNet-S 400 MB, TransPathNet-B 1.3 GB) |
| `training_curve.png` | 왼쪽 손실 곡선(Train, Val), 오른쪽 검증 RMSE·MAE(좌축)와 SSIM(우축, 점선). 정규화 단위 |
| `training_complete.txt` | `model=`, `freq=`, `seed=`, `epochs=`, `batch_size=`, `accum_steps=`, `amp=`, `best_val_loss=` 한 줄씩. 학습이 끝까지 돌았다는 표시(반복 실험 스크립트가 완료 판정에 사용) |

- epoch별 수치는 메모리의 `history`에만 있고 파일로 남지 않는다. 필요하면 실행 로그를 `tee`로 저장한다.

### 5.5 평가 모드

`--mode eval`은 체크포인트를 읽어 seed 42 테스트 분할을 batch 1로 평가한다.

- 샘플마다 `fine` 출력과 정답을 dBm으로 역정규화하고, 건물 마스크 `x[:, 0] < 0.5`인 유효 픽셀에서 RMSE, MAE, bias(예측 − 정답의 평균)를 계산한다(`compute_metrics_dbm`).
- SSIM은 정규화 맵 전체(건물 포함)에서 계산한다. 따라서 SSIM만 건물 픽셀의 영향을 받는다.
- 앞 `--n_vis`개 샘플에 대해 `result_<순번 3자리>_sample<번호>.png`를 저장한다. 5개 판: 건물 맵, 정답, coarse, fine(jet, -90 ~ 0 dBm, 건물 흰색), 잔차(fine − 정답, ±20 dB). 제목에 샘플 번호와 RMSE·MAE·SSIM.
- 끝에 테스트 샘플 평균 `Test RMSE`, `Test MAE`, `Test Bias`, `Test SSIM`을 출력한다. CSV로 저장하지 않는다. 표 형태의 지표가 필요하면 6.3의 스크립트를 쓴다.
- `[Eval] ... (epoch N)`의 N은 `best.pth`가 저장된 epoch이다.

### 5.6 난수와 재현성

- `--seed`는 분할(`randperm`), 학습 로더 셔플(`seed + 1000`), worker 증강(`seed + worker_id`), 가중치 초기화에 모두 쓰인다.
- 같은 seed로 다시 학습해도 cuDNN 합성곱의 비결정적 알고리즘과 AMP 때문에 가중치가 비트 단위로 같지 않다. `--deterministic`은 cuDNN을 결정적 모드로 바꾸지만 느려지며 일부 연산은 여전히 비결정적이다.
- 평가는 결정적이다. 같은 체크포인트·데이터·소프트웨어 버전이면 결과가 같고, GPU 아키텍처가 다르면 TransPathNet의 어텐션 연산 순서 차이로 소수점 6째 자리 이하가 달라진다. RadioUNet은 관측된 범위에서 동일했다(8.3).

## 6. 분석 스크립트와 출력 지표

### 6.1 파이프라인 구성

`scripts/run_1750_4750_5750_shared_pipeline.sh`는 아래 네 스크립트를 순서대로 실행한다. 변수는 스크립트 상단에 있으며 `ROOT`, `CKPT_DIR`, `OUT_DIR`만 환경 변수로 바꿀 수 있다.

| 변수 | 값 | 의미 |
|---|---|---|
| `ROOT` | `<패키지 루트>/data`(환경 변수로 변경 가능) | 데이터셋 루트 |
| `CKPT_DIR` | `<패키지 루트>/model_checkpoints`(변경 가능) | 체크포인트 루트 |
| `OUT_DIR` | `<패키지 루트>/results_1750_4750_5750_regenerated`(변경 가능) | 출력 루트. 아래에 `metrics/`, `miss_cdf/`, `boundary_cdf/`, `qualitative/` |
| `FREQS` | `1750 4750 5750` | 주파수 |
| `THRESHOLDS` | `-70 -65 -60` | 고정 임계값(dBm) |
| `QUAL_THRESHOLD` | `-65` | 정성 그림의 샘플 선택·표시 임계값 |
| `SAMPLE_ID` | `auto` | 정성 그림의 공통 샘플. `auto`면 자동 선택(6.6) |
| `BOUNDARY_WIDTH` | `10` | 경계 영역 폭(dB) |
| `NUM_WORKERS` | `4` | DataLoader worker 수 |

| 단계 | 스크립트 | 입력 | 출력 폴더 |
|---|---|---|---|
| [1/4] 고정 임계값 지표 | `final_eval_metrics_fixed_threshold_heatmap.py` | 데이터셋, 체크포인트 12개 | `metrics/` |
| [2/4] 샘플별 miss rate CDF | `final_samplewise_cdf_v3.py` | `metrics/per_sample_metrics_fixed_threshold.csv` | `miss_cdf/` |
| [3/4] 경계 영역 CDF | `boundary_only_cdf_analysis_x10.py` | 데이터셋, 체크포인트 | `boundary_cdf/` |
| [4/4] 정성 그림 | `qualitative_shared_sample_1750_4750_5750.py` | 데이터셋, 체크포인트, [1/4]의 per-sample CSV | `qualitative/` |

- [1/4]가 끝난 뒤 `per_sample_metrics_fixed_threshold.csv`가 없으면 중단한다(`set -e`). [1/4]와 [4/4]는 체크포인트가 하나라도 없으면 `FileNotFoundError`로 멈추고, [3/4]는 없는 모델을 건너뛴다.
- 각 스크립트는 `scripts/`를 작업 디렉터리로 실행되며 `--seed`를 넘기지 않으므로 분할 seed는 42이다.

### 6.2 공통 규칙

모델을 실행하는 스크립트([1/4], [3/4], [4/4]와 `final_qualitative_split_allpairs_v11.py`)가 공유하는 규칙이다.

| 항목 | 규칙 |
|---|---|
| 유효 픽셀 | `valid = building < 0.5`(입력 0번 채널). 지표는 유효 픽셀에서만 계산 |
| 음영 판정 | `pred_shadow = (pred_dbm < T) & valid`, `gt_shadow = (gt_dbm < T) & valid`. T는 임계값(dBm) |
| 혼동 행렬 | `TP = pred & gt`, `FP = pred & ~gt`, `FN = ~pred & gt`, `TN = ~pred & ~gt & valid`(픽셀 수) |
| 모델 출력 | `(coarse, fine, aux)` 중 `fine`. 역정규화 `denorm(x) = x × 90 − 90` |
| 체크포인트 선택 | `resolve_ckpt_path(ckpt_dir, tag, freq)`: `checkpoints_<tag의 -를 _로>_<freq>/best.pth`. 단 `radiounet-small`, 4750 MHz는 `_v2`가 있으면 우선 |
| 모델 표시 이름 | `RadioUNet-S (35M)`, `RadioUNet-M (97M)`, `TransPathNet-S (38M)`, `TransPathNet-B (111M)`(CSV `model_label`) |
| 임계값 표기 | `threshold_tag(-65.0) = "-65"`(정수) → 파일명 `..._th-65...`. 정수가 아니면 `.`를 `p`, `-`를 `m`으로 |
| 테스트 로더 | `get_dataloaders(..., batch_size=1)`의 세 번째 반환값. 샘플 순서는 seed 42의 `randperm` 순서(`sample_rank_in_test_loader`) |

### 6.3 고정 임계값 지표(final_eval_metrics_fixed_threshold_heatmap.py)

주파수·모델·임계값마다 테스트 샘플별 음영 검출 지표를 계산하고 요약 CSV, heatmap, 표 그림을 만든다. `compute_shadow_metrics`의 정의는 다음과 같다(분모에 1e-8을 더함).

| 지표 | 정의 | 비고 |
|---|---|---|
| `precision` | TP / (TP + FP) | 예측 음영 중 실제 음영 비율 |
| `recall` | TP / (TP + FN) | 실제 음영 중 검출한 비율 |
| `iou` | TP / (TP + FP + FN) |  |
| `f1` | 2·P·R / (P + R) |  |
| `miss_rate` | FN / (TP + FN) | = 1 − recall. 놓친 음영 비율 |
| `false_alarm_rate` | FP / (FP + TN) | 비음영을 음영으로 판정한 비율 |
| `gt_shadow_ratio`, `pred_shadow_ratio` | 음영 픽셀 수 / 유효 픽셀 수 | 정답·예측 각각 |
| `rmse_dbm`, `mae_dbm` | 유효 픽셀의 (pred − gt) RMSE, MAE | 임계값과 무관 |

- 요약(`summarize`): (임계값, 주파수, 모델)마다 샘플별 값의 평균·표준편차(`<지표>_mean`, `<지표>_std`, 모집단 표준편차)와, 픽셀 수를 모두 더해 계산한 micro 지표(`micro_TP` 등, `micro_precision`, `micro_recall`, `micro_iou`, `micro_f1`)를 함께 기록한다. heatmap과 표는 샘플별 평균(macro)을 쓴다.

| 출력 파일 | 내용 |
|---|---|
| `per_sample_metrics_fixed_threshold.csv` | 샘플 × 모델 × 임계값 행. 컬럼은 부록 B.1 |
| `summary_metrics_fixed_threshold.csv` | (임계값, 주파수, 모델) 요약. 컬럼은 부록 B.2 |
| `heatmap_fixed_threshold_th<T>.png` | 2×2(Precision, Recall, IoU, F1). 행 주파수, 열 모델, 값 표기, 행별 최고 굵게 |
| `table_fixed_threshold_th<T>.png` | `Freq., Model, Precision, Recall, IoU, F1-score`(mean ± std) 표 그림 |
| `README_fixed_threshold_metrics.txt` | 정의 요약과 `split seed` |

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--root`(필수), `--ckpt_dir` | –, `./model_checkpoints` | 데이터셋, 체크포인트 루트 |
| `--result_dir` | `./final_results/fixed_threshold_metrics_heatmap` | 출력 폴더 |
| `--freqs` | `1750 2750 3750 4750 5750` | 주파수 목록 |
| `--thresholds` | `-70 -65 -60` | 임계값 목록(dBm) |
| `--models` | 4개 태그 전부 | 모델 목록 |
| `--img_size`, `--num_workers`, `--batch_size` | `256`, `4`, `1` | 크기, worker 수, batch(테스트 로더는 1 고정) |
| `--seed` | `42` | 분할 seed |
| `--min_dbm`, `--max_dbm` | `-90`, `0` | 정규화 범위 |
| `--heatmap_vmin`, `--heatmap_vmax`, `--cmap`, `--dpi` | `0`, `1`, `viridis`, `200` | heatmap 색 범위, 컬러맵, 해상도 |

### 6.4 샘플별 miss rate CDF(final_samplewise_cdf_v3.py)

6.3의 per-sample CSV만 읽어(모델 추론 없음) 임계값·주파수마다 네 모델의 샘플별 miss rate 경험적 CDF를 그린다.

- `miss_rate = FN / (TP + FN)`(분모 0이면 NaN으로 제외). CDF 값 `cdf_fraction(x) = mean(miss_rate ≤ x)`를 `--x_step`(0.0025) 간격의 x 격자에서 계산한다.
- 주 그림 `fig_cdf_missrate_main_th<T>.png`: 1 × `--main_freqs` 개 판. 판 제목에 `Δmean`(네 모델 평균 miss rate의 최대 − 최소)을 적는다.
- 확대 그림 `fig_cdf_missrate_zoom_<주파수>MHz_th<T>.png`: `choose_zoom_window`가 네 모델 CDF의 차이가 가장 큰 x를 찾아 ±`--zoom_half_width`(0.07) 구간과 그 안의 y 범위 ±`--zoom_margin_y`(0.06)를 창으로 잡는다. 창 정보는 `selected_zoom_windows_v3.csv`에 기록한다.
- 선 스타일: RadioUNet-S 파랑 실선, RadioUNet-M 주황 실선, TransPathNet-S 초록 점선, TransPathNet-B 빨강 점선.

| 출력 파일 | 내용 |
|---|---|
| `samplewise_error_rates_v3.csv` | 샘플별 miss rate, false alarm rate와 TP/FP/FN/TN(부록 B.4) |
| `cdf_summary_v3.csv` | 임계값·주파수별 모델 평균 miss rate와 `gap_mean_miss_rate` |
| `fig_cdf_missrate_main_th<T>.png`, `fig_cdf_missrate_zoom_<f>MHz_th<T>.png` | 주 그림, 확대 그림 |
| `selected_zoom_windows_v3.csv`, `README_v3.txt` | 확대 창, 정의 요약 |

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--input_csv`(필수) | – | 6.3의 per-sample CSV |
| `--result_dir` | `./final_results/cdf_v3` | 출력 폴더 |
| `--freqs`, `--main_freqs`, `--zoom_freqs` | 5개 주파수, `1750 3750 5750`, `1750` | 처리·주 그림·확대 주파수(파이프라인은 `1750 4750 5750`, `1750 4750 5750`, `1750`을 넘김) |
| `--thresholds`, `--threshold_tol` | `-70 -65 -60`, `1e-6` | CSV에 있는 임계값만 처리 |
| `--x_max`, `--x_step`, `--tick_step` | `1.0`, `0.0025`, `0.10` | x 범위·간격·눈금 |
| `--zoom_half_width`, `--zoom_margin_y`, `--dpi` | `0.07`, `0.06`, `180` | 확대 창, 해상도 |

### 6.5 경계 영역 CDF(boundary_only_cdf_analysis_x10.py)

임계값 부근 픽셀만 따로 평가한다. 경계 영역은 `boundary = valid & (|gt_dbm − T| ≤ boundary_width_dbm)`이며 기본 폭은 ±10 dB이다.

| 지표 | 정의 |
|---|---|
| `boundary_mae_db`, `boundary_rmse_db` | 경계 픽셀의 \|pred − gt\| 평균, 제곱평균제곱근 |
| `boundary_bias_db`, `boundary_abs_p90_db` | 경계 픽셀의 (pred − gt) 평균, \|오차\|의 90 백분위 |
| `boundary_miss_rate` | 경계 안 FN / 경계 안 정답 음영 픽셀 수(0이면 NaN) |
| `boundary_false_alarm_rate` | 경계 안 FP / 경계 안 정답 비음영 픽셀 수(0이면 NaN) |
| `boundary_precision`, `_recall`, `_f1`, `_iou` | 경계 안 TP/FP/FN으로 계산 |

- 샘플별 값의 경험적 CDF(`empirical_cdf`, `searchsorted` 기반)를 오차 지표(`--error_metric`, 기본 `boundary_mae_db`)와 miss rate에 대해 그린다. 요약 CSV에는 평균·중앙값·p90과 `F_le_<c>`(값이 c 이하인 샘플 비율, 오차 1/2/5 dB, miss rate 0.05/0.10/0.20)를 기록한다.
- `--threshold_mode per_freq`를 주면 주파수별 임계값 `FREQ_THRESHOLD = {1750: -61.7, 2750: -64.9, 3750: -67.4, 4750: -69.5, 5750: -71.1}`을 쓴다(논문 초기 분석용). 파이프라인은 `fixed`를 쓴다.
- `boundary_only_cdf_analysis.py`(x10 아님)는 지표가 같고 `--error_x_max` 기본값(12 대 10), 글꼴 크기, 축 이름(`Boundary RSS Error` 대 `Threshold-margin RSS Error`), 파일명의 `_x10` 접미사만 다르다.

| 출력 파일 | 내용 |
|---|---|
| `per_sample_boundary_metrics.csv` | 샘플 × 모델 × 임계값 행(부록 B.3) |
| `boundary_cdf_summary.csv` | (주파수, 임계값, 모델) 요약 |
| `fig_boundary_error_cdf_x10_th<T>_bw<W>.png` | 오차 CDF(1 × 주파수 판) |
| `fig_boundary_missrate_cdf_th<T>_bw<W>.png` | miss rate CDF |
| `fig_boundary_combined_cdf_x10_th<T>_bw<W>.png` | 위 둘을 2행으로 합친 그림 |
| `README_boundary_cdf.txt` | 정의 요약 |

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--root`(필수), `--ckpt_dir`, `--result_dir` | –, `./model_checkpoints`, `./final_results/boundary_cdf` | 경로 |
| `--freqs`, `--models` | `1750 3750 5750`, 4개 모델 | 파이프라인은 `1750 4750 5750` |
| `--threshold_mode`, `--thresholds`, `--threshold_tol` | `fixed`, `-65`, `1e-6` | 파이프라인은 `-70 -65 -60` |
| `--boundary_width_dbm` | `10.0` | 경계 폭 |
| `--error_metric` | `boundary_mae_db` | `boundary_rmse_db` 선택 가능 |
| `--error_x_max`, `--error_x_step`, `--miss_x_max`, `--miss_x_step` | `10`, `0.05`, `1.0`, `0.0025` | CDF x 축 범위·간격 |
| `--summary_error_cutoffs`, `--summary_miss_cutoffs` | `1 2 5`, `0.05 0.10 0.20` | 요약의 `F_le` 기준값 |
| `--img_size`, `--num_workers`, `--min_dbm`, `--max_dbm`, `--dpi` | `256`, `4`, `-90`, `0`, `200` |  |

### 6.6 정성 그림(qualitative_shared_sample_1750_4750_5750.py)

세 주파수에 공통으로 존재하는 테스트 샘플 하나를 골라 네 모델의 오차 맵과 음영 검출 결과(TP/FP/FN)를 그린다.

- 샘플 선택(`choose_shared_sample`): 6.3의 per-sample CSV에서 `--selection_threshold`(-65 dBm) 행만 모아, 요청한 모든 주파수에 네 모델 결과가 있는 샘플을 후보로 한다. `--selection_criterion` 점수가 가장 높은 샘플을 고른다.
  - `rm_minus_ts_recall_gap`(기본): 주파수 평균 recall(RadioUNet-M) − recall(TransPathNet-S)
  - `small_pair_recall_gap`: recall(RadioUNet-S) − recall(TransPathNet-S)
  - `average_model_gap`: CNN 두 모델 평균 recall − Transformer 두 모델 평균 recall
- `--sample_id`에 숫자를 주면 후보에 있을 때만 쓰고, 없으면 경고 후 자동 선택한다. 기준 결과의 공통 샘플은 770이다.
- `find_sample`은 테스트 로더를 순서대로 돌며 샘플 번호가 맞는 것을 찾는다. 없으면 `RuntimeError`.
- 색: TP 초록, FP 빨강, FN 파랑, 건물·제외 회색, 그 외 흰색. 오차 맵은 |pred − gt|를 흰색 → 빨강(최대 `--error_vmax` 20 dB)으로 그린다. 모델 배치는 2×2(왼쪽 위 RadioUNet-S, 오른쪽 위 TransPathNet-S, 아래 RadioUNet-M, TransPathNet-B).

| 출력 파일 | 내용 |
|---|---|
| `selected_shared_sample_candidates.csv` | 상위 30개 후보와 점수, 주파수별 recall·gap |
| `selected_shared_sample.csv` | 선택 샘플의 주파수별 네 모델 precision/recall/iou/f1/TP/FP/FN |
| `error_map_4models_<f>MHz_sample<id>_th<T>.png` | 주파수별 2×2 오차 맵(제목에 평균·최대 오차) |
| `shadow_detection_4models_<f>MHz_sample<id>_th<T>.png` | 주파수별 2×2 음영 검출(제목에 TP/FP/FN, recall, F1) |
| `error_map_allfreqs_sample<id>_th<T>.png`, `shadow_detection_allfreqs_sample<id>_th<T>.png` | 행 주파수 × 열 모델 비교 그림 |

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--root`(필수), `--ckpt_dir`, `--metrics_csv`(필수), `--result_dir` | –, `./model_checkpoints`, –, `./results_1750_4750_5750/qualitative` | 경로 |
| `--freqs` | `1750 4750 5750` | 주파수 |
| `--sample_id` | `auto` | 공통 샘플 번호 또는 자동 |
| `--selection_threshold`, `--shadow_threshold` | `-65`, `-65` | 선택용·표시용 임계값 |
| `--selection_criterion` | `rm_minus_ts_recall_gap` | 선택 기준 |
| `--error_vmax`, `--dpi`, `--img_size`, `--num_workers`, `--min_dbm`, `--max_dbm` | `20`, `180`, `256`, `4`, `-90`, `0` |  |

### 6.7 반복 holdout 실험

`scripts/run_multi_seed_holdout_experiments.sh`는 분할 seed마다 학습(TRAIN) → 고정 임계값 지표(EVAL) → 집계(AGGREGATE)를 수행한다. 설정은 환경 변수로 준다(『설치 매뉴얼』 5.5의 표).

- 학습 명령: `python -m ikeee.bratlab_train --mode train ... --save_dir <OUT_DIR>/seed_<seed>/model_checkpoints/checkpoints_<모델>_<크기>_<주파수> --seed <seed>`. TransPathNet 모델에는 `TRANSPATHNET_BATCH_SIZE`, `TRANSPATHNET_ACCUM_STEPS`, `TRANSPATHNET_AMP`를 적용한다.
- `SKIP_EXISTING=1`이면 `best.pth`와 `training_complete.txt`(또는 `training_curve.png`)가 있는 조합을 건너뛴다.
- 평가 명령: `final_eval_metrics_fixed_threshold_heatmap.py --seed <seed>`로 seed별 분할에서 지표를 계산해 `seed_<seed>/metrics/`에 저장한다. 경계 영역·정성 스크립트는 `--seed`가 없어 반복 실험에 포함되지 않는다.
- 집계(`aggregate_seed_metrics.py`): `seed_*/metrics/summary_metrics_fixed_threshold.csv`를 모아 `summary_metrics_by_seed.csv`, `summary_metrics_across_seeds.csv`(seed 평균·표본 표준편차, `--ddof 1`), `paper_table_th<T>.csv`(`mean +/- std` 문자열)를 `aggregate/`에 쓴다. CSV에 `seed` 열이 없으면 폴더 이름 `seed_<n>`에서 읽는다.
- `resume_seed43_44_remaining.sh`: 기준 결과의 seed 42 지표를 `seed_42/metrics/`로 복사한 뒤 seed 43, 44의 남은 학습·평가·집계를 수행한다. TransPathNet 설정은 호출자가 환경 변수로 넘겨야 한다.

### 6.8 보조 스크립트

| 스크립트 | 입력 | 출력 | 용도 |
|---|---|---|---|
| `check_setup.py` | `--root`, `--ckpt_dir`, `--freqs` | 화면 출력, 종료 코드 0/1 | 패키지·GPU·코드·데이터셋 점검(OK/FAIL), 체크포인트 표(참고) |
| `final_qualitative_split_allpairs_v11.py` | 데이터셋, 체크포인트, `--freq`(3750), `--sample_id`(1486) | `qualitative_errormap_allpairs_*.png`, `qualitative_shadow_allpairs_*.png` | 한 주파수·한 샘플의 오차 맵과 GT 기준 TP/FN 그림(결과 9) |
| `paper_missrate_cdf_builder.py` | 6.3의 per-sample CSV | `samplewise_miss_rate_rows.csv`, `cdf_summary_table.csv`, `cdf_curve_points.csv`, `fig_missrate_cdf_*.png` | 5개 주파수 miss rate CDF와 요약 표(결과 7). 파일명 임계값이 `th-65.0` 형식 |
| `plot_existing_cdf_font_plus10.py` | `results/representative_.../metrics`, `boundary_cdf`의 CSV | `*_font_plus10/` 그림 | 논문용 큰 글꼴 재작도(결과 5) |
| `plot_fixed_threshold_heatmap_font_plus10.py` | `summary_metrics_fixed_threshold.csv` | `fixed_threshold_th-65.png`, `heatmap_..._font_plus10.png` | -65 dBm heatmap 큰 글꼴(결과 6) |
| `plot_rmse_vs_frequency_paper.py` | 요약 CSV(`--summary_csv`, `rmse_dbm_mean/std` 필요) | `fig_rmse_vs_frequency.png` | 주파수별 RMSE 오차막대. 기본 입력 경로가 패키지에 없으므로 `--summary_csv`, `--out` 지정 |
| `boundary_only_cdf_analysis.py` | 6.5와 같음 | x축 12 dB 판 그림 | 결과 8의 `fig_boundary_*_th-65_bw10.png`(x10 없는 파일) |

### 6.9 결과 해석 예

기준 결과 `results/representative_1750_4750_5750/metrics/summary_metrics_fixed_threshold.csv`의 -65 dBm 행(샘플별 평균)이다.

| 주파수 | 모델 | Precision | Recall | IoU | F1 | Miss rate | RMSE(dBm) | MAE(dBm) |
|---|---|---|---|---|---|---|---|---|
| 1750 | RadioUNet-S | 0.886 | 0.850 | 0.764 | 0.862 | 0.150 | 2.193 | 1.574 |
| 1750 | RadioUNet-M | 0.876 | 0.874 | 0.776 | 0.870 | 0.126 | 2.088 | 1.490 |
| 1750 | TransPathNet-S | 0.869 | 0.844 | 0.745 | 0.850 | 0.156 | 2.475 | 1.810 |
| 1750 | TransPathNet-B | 0.880 | 0.861 | 0.769 | 0.866 | 0.139 | 2.254 | 1.593 |
| 4750 | RadioUNet-S | 0.908 | 0.898 | 0.824 | 0.901 | 0.102 | 2.466 | 1.792 |
| 4750 | RadioUNet-M | 0.920 | 0.922 | 0.853 | 0.919 | 0.078 | 2.064 | 1.463 |
| 4750 | TransPathNet-S | 0.934 | 0.867 | 0.817 | 0.897 | 0.133 | 2.509 | 1.784 |
| 4750 | TransPathNet-B | 0.919 | 0.915 | 0.847 | 0.915 | 0.085 | 2.268 | 1.548 |
| 5750 | RadioUNet-S | 0.917 | 0.931 | 0.860 | 0.923 | 0.069 | 2.196 | 1.575 |
| 5750 | RadioUNet-M | 0.924 | 0.949 | 0.881 | 0.936 | 0.051 | 1.951 | 1.373 |
| 5750 | TransPathNet-S | 0.928 | 0.899 | 0.841 | 0.912 | 0.101 | 2.506 | 1.774 |
| 5750 | TransPathNet-B | 0.920 | 0.931 | 0.862 | 0.924 | 0.069 | 2.165 | 1.537 |

- 정답 음영 비율(`gt_shadow_ratio_mean`)은 1750 MHz 0.217, 4750 MHz 0.457, 5750 MHz 0.505로 주파수가 높을수록 음영이 넓고, 지표도 높아진다. 주파수 간 비교는 음영 비율 차이를 함께 본다.
- RMSE·MAE는 임계값과 무관하므로 -70, -60 dBm 행에도 같은 값이 들어 있다.
- 같은 크기급에서 CNN(RadioUNet-M)이 Transformer(TransPathNet-B)보다 recall이 높고 miss rate가 낮다. TransPathNet-S는 precision이 높지만 recall이 낮아 음영을 보수적으로 검출한다. 경계 영역(6.5)에서는 RadioUNet-M의 MAE가 세 주파수 모두 가장 낮다(1.61, 1.51, 1.39 dB).
- miss rate CDF의 `gap_mean_miss_rate`(네 모델 평균 miss rate의 최대 − 최소)는 -65 dBm에서 1750 MHz 0.030, 4750 MHz 0.055, 5750 MHz 0.050이다.

## 7. 코드 수정과 확장

### 7.1 수정 위치

| 바꾸려는 것 | 위치 |
|---|---|
| 입력 채널(송신기 수, 추가 입력) | `bratlab_dataset.BRATLabDataset.__getitem__`(채널 생성)과 `load_stations`(열 수). 모델의 입력 채널은 `RadioUNet(in_ch)`, `TransPathNetOutdoor`의 `TransPathUnit(in_ch=2/3)`, `build_*` |
| 정규화 범위(-90 ~ 0 dBm) | `BRATLabDataset(min_dbm, max_dbm)`, `bratlab_train.denorm`, 분석 스크립트의 `--min_dbm`, `--max_dbm`. 학습·평가·분석에서 모두 같은 값 |
| 입출력 크기(256) | `--img_size`(학습, 분석 스크립트). TransNeXt 백본과 EMCAD 디코더는 4단계 축소·확대 구조이므로 32의 배수를 전제한다 |
| 분할 비율·seed | `get_dataloaders(split=(0.7, 0.1, 0.2), seed)`. 비율을 바꾸면 기준 결과와 비교 불가 |
| 증강 | `BRATLabDataset.__getitem__`의 `if self.augment:` 블록 |
| 모델 크기 | `radiounet.MODEL_CONFIGS`, `transpathnet_outdoor.MODEL_CONFIGS`. 새 크기 이름은 `bratlab_train.build_model`의 assert와 `--size` choices에도 추가 |
| 손실 가중치 | `RadioLoss(w_mse, w_ssim, w_aux)` 생성 인자와 `train`의 `0.4 *`(coarse 비중) |
| 옵티마이저·스케줄 | `train`의 `AdamW(weight_decay=1e-4)`, `CosineAnnealingLR(eta_min=1e-6)`, `clip_grad_norm_(1.0)` |
| 임계값·경계 폭·주파수(파이프라인) | `run_1750_4750_5750_shared_pipeline.sh`의 `THRESHOLDS`, `BOUNDARY_WIDTH`, `FREQS`, `QUAL_THRESHOLD`. 개별 스크립트는 해당 옵션 |
| 모델 목록·표시 이름 | 각 분석 스크립트의 `MODEL_ORDER`, `MODELS_CONFIG`/`MODEL_LABELS`, `MODEL_STYLE`; `check_setup.MODEL_TAGS`; 반복 실험 스크립트의 `model_to_arch_size` |
| 새 주파수 | `data/receivedpower_<주파수>MHz_mat/` 추가, `bratlab_train --freq` choices, `check_setup.FREQS`, 분석 스크립트 `--freqs` |
| 결과 CSV 컬럼 | 6.3 `evaluate`의 row dict와 `summarize`; 6.4 ~ 6.6은 입력 컬럼 이름(`REQUIRED_COLUMNS` 등)을 함께 수정 |

### 7.2 새 모델 추가 순서

1. `ikeee/model/<이름>.py`에 모델 클래스와 `build_<이름>(size)` 함수를 만든다. `forward(x)`는 `(coarse, fine, aux)`를 돌려주고 각 출력은 (B, 1, H, W), [0, 1] 범위여야 한다. 2단계가 아니면 `coarse = fine`, `aux = fine`으로 돌려준다.
2. `bratlab_train.build_model`에 분기를 추가하고 `--model` choices에 이름을 넣는다. `model_tag`는 `<model>-<size>` 형식이 된다.
3. 분석 스크립트의 `MODELS_CONFIG`(또는 `MODEL_ORDER`)와 표시 이름·선 스타일에 태그를 추가한다. 6.3, 6.5, 6.6, `final_qualitative_split_allpairs_v11.py`는 각각 `load_model`이 생성 함수를 호출하므로 import와 분기를 추가한다.
4. `check_setup.MODEL_TAGS`와 `run_multi_seed_holdout_experiments.sh`의 `model_to_arch_size`에 추가한다.
5. 학습 후 체크포인트를 `model_checkpoints/checkpoints_<태그의 -를 _로>_<주파수>/best.pth`에 두면 파이프라인이 자동으로 찾는다.

### 7.3 실습 예제: 임계값 -75 dBm 추가

고정 임계값에 -75 dBm을 추가해 지표와 miss rate CDF를 다시 만드는 예이다. 코드를 바꾸지 않고 옵션만으로 수행한다.

**실행 위치**: 호스트 · 작업 디렉터리: 패키지 루트(컨테이너 명령 접두어는 『설치 매뉴얼』 5.1)

```bash
docker compose run --rm rmp bash -c "cd scripts && python final_eval_metrics_fixed_threshold_heatmap.py \
    --root /workspace/data --ckpt_dir /workspace/model_checkpoints \
    --result_dir /workspace/runs/metrics_th75 --freqs 1750 4750 5750 --thresholds -75 -70 -65 -60"
docker compose run --rm rmp bash -c "cd scripts && python final_samplewise_cdf_v3.py \
    --input_csv /workspace/runs/metrics_th75/per_sample_metrics_fixed_threshold.csv \
    --result_dir /workspace/runs/miss_cdf_th75 --freqs 1750 4750 5750 --main_freqs 1750 4750 5750 --thresholds -75 -70 -65 -60"
ls runs/metrics_th75 runs/miss_cdf_th75
```

정상 출력 예

```text
runs/metrics_th75:
README_fixed_threshold_metrics.txt  heatmap_fixed_threshold_th-60.png  heatmap_fixed_threshold_th-65.png  heatmap_fixed_threshold_th-70.png  heatmap_fixed_threshold_th-75.png ...
runs/miss_cdf_th75:
README_v3.txt  cdf_summary_v3.csv  fig_cdf_missrate_main_th-60.png ... fig_cdf_missrate_main_th-75.png  fig_cdf_missrate_zoom_1750MHz_th-75.png ...
```

- 파이프라인에 영구히 반영하려면 `run_1750_4750_5750_shared_pipeline.sh`의 `THRESHOLDS="-75 -70 -65 -60"`으로 바꾼다. 경계 영역 단계와 정성 그림의 표시 임계값은 각각 `--thresholds`, `QUAL_THRESHOLD`를 따른다.
- 스크립트 파일은 `scripts/`를 작업 디렉터리로 실행해야 상대 import가 맞으므로 `cd scripts`를 붙였다. 경로는 컨테이너 기준(`/workspace/...`)으로 준다.

### 7.4 변경 사항 점검

**실행 위치**: 호스트 · 작업 디렉터리: 패키지 루트

```bash
docker compose run --rm rmp python scripts/check_setup.py
docker compose run --rm rmp python -m ikeee.bratlab_train --mode train --root data --freq 4750 --model radiounet --size small \
    --batch_size 4 --amp --epochs 1 --save_dir runs/smoke_train
docker compose run --rm rmp python -m ikeee.bratlab_train --mode eval --root data --freq 4750 --model radiounet --size small \
    --ckpt model_checkpoints/checkpoints_radiounet_small_4750_v2/best.pth --result_dir runs/smoke_eval --n_vis 2
docker compose run --rm -e OUT_DIR=/workspace/runs/pipeline_check rmp bash scripts/run_1750_4750_5750_shared_pipeline.sh
diff <(cut -d, -f2- runs/pipeline_check/metrics/summary_metrics_fixed_threshold.csv) results/representative_1750_4750_5750/metrics/summary_metrics_fixed_threshold.csv
```

- 순서대로 import·환경, 학습 루프, 평가, 파이프라인 전체, 기준 결과와의 비교를 확인한다. 모델·손실·데이터 로더를 바꾸지 않았다면 마지막 diff는 TransPathNet 행의 소수점 6째 자리 이하 차이만 보여야 한다.
- 모델이나 데이터 처리를 바꾼 경우에는 기준 결과와 같을 수 없으므로, 변경 전 코드로 같은 환경에서 결과를 만들어 비교 기준으로 쓴다.

## 8. 제약사항과 유의사항

### 8.1 모델 범위

- 입력은 건물 유무와 송신기 3개 위치뿐이다(1.4). 송신기 출력, 안테나, 주파수는 입력에 없고 주파수별 체크포인트로 구분한다.
- 256 × 256으로 축소한 라디오맵에서 학습·평가하므로 원본 해상도의 세부 구조는 보간으로 평활화된다.
- 2단계 구조의 fine 단계는 coarse 출력을 `detach()`해 받으므로 coarse 단계는 자신의 손실로만 학습된다.

### 8.2 결과 해석 유의

- 학습 로그와 `training_curve.png`의 RMSE·MAE는 정규화 단위([0, 1], 건물 포함)이고, 평가 모드와 분석 스크립트의 RMSE·MAE는 dBm 단위(건물 제외)이다. 두 값을 직접 비교하지 않는다.
- 평가 모드의 SSIM은 건물을 포함한 전체 맵에서 계산한다.
- 요약 CSV의 `<지표>_mean`은 샘플별 평균(macro), `micro_*`는 픽셀 수 합산 기준이다. 음영이 넓은 샘플의 비중이 다르므로 두 값이 다르다. 논문과 heatmap은 macro를 쓴다.
- 기준 결과의 지표 CSV에는 `seed` 열이 없고 재현 CSV에는 있다. 비교할 때 첫 열을 제외한다. 기준 결과의 일부 그림은 이전 판 스크립트로 만들어 글꼴·배치가 현재 출력과 다를 수 있다.
- `results/samplewise_cdf/samplewise_error_rates.csv`는 주파수별 임계값(-61.7 ~ -71.1 dBm)으로 만든 초기 분석 파일이며 현재 스크립트로는 생성되지 않는다.
- 분석 스크립트의 기본 주파수·임계값은 파이프라인과 다르다(6.4 `--main_freqs` 기본 `1750 3750 5750`, 6.5 `--freqs` 기본 `1750 3750 5750`, `--thresholds` 기본 `-65`). 직접 실행할 때는 옵션을 명시한다.
- 4750 MHz RadioUNet-S의 분석 결과는 `_v2` 체크포인트(lr 3e-5)로 만든 것이다. `checkpoints_radiounet_small_4750/best.pth`로 단일 평가하면 값이 다르다.

### 8.3 재현성

같은 체크포인트·데이터셋·환경 이미지에서 평가와 분석은 결정적이다. GPU가 다르면 TransPathNet의 결과가 소수점 6째 자리 이하에서 달라지며, RadioUNet은 동일했다. 학습은 같은 seed라도 비트 단위로 재현되지 않는다.

- Windows 11 + GTX 1050(환경 이미지)에서 대표 결과를 재현한 결과, 36개 요약 행 중 RadioUNet 18행은 기준 결과(RTX 2080 Ti)와 모든 자리에서 같았고 TransPathNet 18행은 1e-6 수준에서만 달랐다(『설치 매뉴얼』 부록 B.4).
- 정성 그림의 공통 샘플 선택은 recall 차이의 순위로 정해지므로 1e-6 수준 차이로 바뀌지 않았다(770).
- 학습 재현에는 `--seed`와 `--deterministic`을 쓰되, 체크포인트가 같지 않을 수 있음을 전제로 지표의 seed 평균·표준편차(6.7)로 해석한다.
- `pytorch_msssim`이 없는 환경에서 학습하면 손실이 달라져 다른 모델이 된다.

### 8.4 실행 도구

- `ikeee`는 패키지 루트에서 `python -m`으로 실행하거나 `PYTHONPATH`에 루트가 있어야 한다. 환경 이미지는 `PYTHONPATH=/workspace`로 설정되어 있다.
- `scripts/`의 파이썬 파일은 `scripts/`에서 실행하는 것을 전제로 한다(파이프라인 셸이 `cd "$SCRIPT_DIR"`). 다른 위치에서 실행해도 루트를 찾지만, 상대 경로 옵션은 작업 디렉터리 기준이다.
- `get_dataloaders`는 테스트 로더의 batch size를 1로 고정한다. 6.3의 `--batch_size`는 테스트 평가에 영향이 없다.
- 데이터셋은 컨테이너에 읽기 전용으로 연결된다. 데이터 전처리 결과를 저장하는 코드를 추가하면 `data/` 밖에 써야 한다.
- `torch.cuda.amp.GradScaler`, `autocast` API를 쓴다. PyTorch 2.4 이상에서는 경고가 나오지만 동작한다.
- `bratlab_train.py --save_dir`의 기본값 `./checkpoints`는 `.gitignore`에 있다. 체크포인트는 `model_checkpoints/` 아래 규칙 이름으로 저장한다.

### 8.5 지원 범위

본 패키지는 환경 이미지(PyTorch 2.2.0 + CUDA 11.8) 기준으로 Windows 11 + WSL 2 + Docker Desktop(GTX 1050) 환경에서 검증했으며, 논문 결과는 Ubuntu 20.04 + RTX 2080 Ti에서 같은 버전으로 생성했다. 다른 PyTorch·CUDA 버전에서는 결과가 달라질 수 있고 지원 범위 밖이다(『설치 매뉴얼』 3.5).

## 부록 A. CLI 옵션 전체

5.3(학습·평가), 6.3 ~ 6.6(분석 스크립트)의 옵션 표에 없는 스크립트의 옵션이다.

### A.1 check_setup.py

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--root` | `<패키지 루트>/data` | 데이터셋 루트 |
| `--ckpt_dir` | `<패키지 루트>/model_checkpoints` | 체크포인트 루트 |
| `--freqs` | `1750 2750 3750 4750 5750` | 점검할 주파수. 하나라도 파일이 없으면 데이터셋 MISSING |

### A.2 aggregate_seed_metrics.py

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--result_root`(필수) | – | `seed_*` 폴더가 있는 루트 |
| `--summary_glob` | `seed_*/metrics/summary_metrics_fixed_threshold.csv` | 요약 CSV 패턴 |
| `--output_dir` | `<result_root>/aggregate` | 출력 폴더 |
| `--ddof` | `1` | seed 표준편차의 자유도 보정(1 = 표본 표준편차) |

### A.3 run_multi_seed_holdout_experiments.sh 환경 변수

| 변수 | 기본값 | 의미 |
|---|---|---|
| `PYTHON` | `python` | 파이썬 실행 파일 |
| `ROOT`, `OUT_DIR` | `<루트>/data`, `<루트>/results_multi_seed_holdout` | 경로 |
| `SEEDS`, `FREQS`, `MODELS`, `THRESHOLDS` | `42 43 44 45 46`, `1750 4750 5750`, 4개 태그, `-70 -65 -60` | 실험 범위 |
| `EPOCHS`, `BATCH_SIZE`, `ACCUM_STEPS`, `AMP`, `LR`, `IMG_SIZE`, `NUM_WORKERS` | `100`, `8`, `1`, `0`, `1e-4`, `256`, `4` | 학습 설정 |
| `TRANSPATHNET_BATCH_SIZE`, `TRANSPATHNET_ACCUM_STEPS`, `TRANSPATHNET_AMP` | `BATCH_SIZE`, `ACCUM_STEPS`, `AMP` | TransPathNet 전용 학습 설정 |
| `TRAIN`, `EVAL`, `AGGREGATE`, `SKIP_EXISTING`, `DETERMINISTIC`, `DRY_RUN` | `1`, `1`, `1`, `1`, `0`, `0` | 단계 선택, 건너뛰기, 결정적 모드, 명령만 출력 |
| `EVAL_SCRIPT`, `AGG_SCRIPT` | `scripts/final_eval_metrics_fixed_threshold_heatmap.py`, `scripts/aggregate_seed_metrics.py` | 사용할 스크립트 |
| `MPLCONFIGDIR`, `PYTORCH_CUDA_ALLOC_CONF` | `/tmp/matplotlib-<USER>`, `expandable_segments:True` | 내보내는 환경 변수 |

### A.4 final_qualitative_split_allpairs_v11.py

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--root`(필수), `--ckpt_dir`, `--result_dir` | –, `./model_checkpoints`, `./final_results/qualitative_split_allpairs_v11` | 경로 |
| `--freq`, `--sample_id` | `3750`, `1486` | 주파수, 샘플 번호(테스트 분할에 있어야 함) |
| `--shadow_threshold` | `-65` | 표시 임계값 |
| `--error_clip_db`, `--error_percentile` | `0.0`, `99.0` | 오차 맵 최댓값(0이면 백분위로 자동) |
| `--img_size`, `--num_workers`, `--min_dbm`, `--max_dbm`, `--dpi` | `256`, `4`, `-90`, `0`, `200` |  |

### A.5 paper_missrate_cdf_builder.py

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--input_csv`(필수), `--result_dir` | –, `./final_results/paper_cdf` | 6.3의 per-sample CSV, 출력 폴더 |
| `--models`, `--freqs`, `--main_freqs`, `--zoom_freqs` | 4개 태그, 5개 주파수, `1750 3750 5750`, `1750` | 범위 |
| `--thresholds`, `--threshold_tol` | `-65`, `1e-6` | 임계값 |
| `--x_max`, `--x_step`, `--x_tick_step` | `1.0`, `0.0025`, `0.1` | x 축 |
| `--zoom_half_width`, `--zoom_y_margin` | `0.06`, `0.05` | 확대 창 |
| `--summary_x`, `--dpi` | `0.05 0.10 0.20`, `200` | 요약 표의 F(x) 기준, 해상도 |

### A.6 그림 재작도 스크립트

| 스크립트 | 주요 옵션(기본값) |
|---|---|
| `plot_existing_cdf_font_plus10.py` | `--metric_csv`, `--boundary_csv`(기준 결과 CSV), `--miss_out_dir`, `--boundary_out_dir`(`*_font_plus10`), `--freqs`(`1750 4750 5750`), `--zoom_freqs`(`1750`), `--thresholds`(`-70 -65 -60`), `--error_x_max`(`10`), `--boundary_width_dbm`(`10`), `--dpi`(`200`) |
| `plot_fixed_threshold_heatmap_font_plus10.py` | `--summary_csv`(`results/fixed_threshold_metrics/summary_metrics_fixed_threshold.csv`), `--threshold`(`-65`), `--freqs`(`1750 4750 5750`), `--models`, `--out`, `--out_copy`, `--vmin`(`0`), `--vmax`(`1`), `--cmap`(`viridis`), `--dpi`(`200`) |
| `plot_rmse_vs_frequency_paper.py` | `--summary_csv`(패키지에 없는 경로가 기본값이므로 지정 필요), `--threshold`(`-65`), `--out`(여러 개 가능), `--dpi`(`180`) |

## 부록 B. 결과 CSV 컬럼

### B.1 per_sample_metrics_fixed_threshold.csv(6.3)

| 컬럼 | 의미 |
|---|---|
| `seed` | 분할 seed(현재 스크립트. 기준 결과 파일에는 없음) |
| `freq_mhz`, `sample_id`, `sample_rank_in_test_loader` | 주파수, 샘플 번호(1 ~ 2000), 테스트 로더 순번(0부터) |
| `model_tag`, `model_label`, `threshold_dbm` | 모델 태그, 표시 이름, 임계값 |
| `TP`, `FP`, `FN`, `TN` | 픽셀 수 |
| `gt_shadow_pixels`, `pred_shadow_pixels`, `valid_pixels` | 정답 음영, 예측 음영, 유효(비건물) 픽셀 수 |
| `gt_shadow_ratio`, `pred_shadow_ratio` | 음영 픽셀 / 유효 픽셀 |
| `precision`, `recall`, `iou`, `f1`, `miss_rate`, `false_alarm_rate` | 6.3의 정의 |
| `rmse_dbm`, `mae_dbm` | 유효 픽셀 RSS 오차(임계값 무관) |

### B.2 summary_metrics_fixed_threshold.csv(6.3)

| 컬럼 | 의미 |
|---|---|
| `seed`, `threshold_dbm`, `freq_mhz`, `model_tag`, `model_label`, `n_samples` | 키와 샘플 수(400, 4750 MHz는 399) |
| `<지표>_mean`, `<지표>_std` | `precision`, `recall`, `iou`, `f1`, `miss_rate`, `false_alarm_rate`, `rmse_dbm`, `mae_dbm`, `gt_shadow_ratio`, `pred_shadow_ratio`의 샘플 평균과 모집단 표준편차 |
| `micro_TP`, `micro_FP`, `micro_FN`, `micro_TN` | 테스트 샘플 전체 픽셀 합 |
| `micro_precision`, `micro_recall`, `micro_iou`, `micro_f1` | 합산 픽셀로 계산한 지표 |

### B.3 per_sample_boundary_metrics.csv, boundary_cdf_summary.csv(6.5)

| 컬럼 | 의미 |
|---|---|
| `freq_mhz`, `sample_id`, `sample_rank_in_test_loader`, `model_tag`, `model_label` | 키 |
| `threshold_mode`, `threshold_dbm`, `boundary_width_dbm` | `fixed`/`per_freq`, 임계값, 경계 폭 |
| `boundary_pixels`, `boundary_gt_shadow_pixels`, `boundary_gt_nonshadow_pixels` | 경계 픽셀 수와 그중 정답 음영·비음영 |
| `boundary_TP`, `boundary_FP`, `boundary_FN`, `boundary_TN` | 경계 안 혼동 행렬 |
| `boundary_mae_db`, `boundary_rmse_db`, `boundary_bias_db`, `boundary_abs_p90_db` | 경계 안 오차 통계 |
| `boundary_miss_rate`, `boundary_false_alarm_rate`, `boundary_precision`, `boundary_recall`, `boundary_f1`, `boundary_iou` | 경계 안 검출 지표 |
| (요약) `mean_boundary_pixels`, `mean_boundary_gt_shadow_pixels` | 샘플 평균 경계 픽셀 수 |
| (요약) `<오차지표>_n/_mean/_median/_p90`, `boundary_miss_rate_n/_mean/_median/_p90` | 유효 샘플 수와 통계 |
| (요약) `<오차지표>_F_le_1/2/5`, `boundary_miss_rate_F_le_0.05/0.1/0.2` | 기준값 이하 샘플 비율 |

### B.4 miss rate CDF 파일(6.4)

| 파일 | 컬럼 |
|---|---|
| `samplewise_error_rates_v3.csv` | `freq_mhz`, `sample_id`, `model_tag`, `model_label`, `threshold_dbm`, `TP`, `FP`, `FN`, `TN`, `miss_rate`, `false_alarm_rate`, `recall`, `precision`, `f1`, `iou`, `rmse_dbm` |
| `cdf_summary_v3.csv` | `threshold_dbm`, `freq_mhz`, `gap_mean_miss_rate`, `radiounet-small_mean_miss_rate`, `radiounet-medium_mean_miss_rate`, `transpathnet-small_mean_miss_rate`, `transpathnet-base_mean_miss_rate` |
| `selected_zoom_windows_v3.csv` | `threshold_dbm`, `freq_mhz`, `x_min`, `x_max`, `y_min`, `y_max` |

### B.5 정성 그림 파일(6.6)

| 파일 | 컬럼 |
|---|---|
| `selected_shared_sample.csv` | `sample_id`, `freq_mhz`, `score`, `criterion`, `selection_threshold_dbm`, 모델 태그마다 `<태그>_precision`, `_recall`, `_iou`, `_f1`, `_TP`, `_FP`, `_FN` |
| `selected_shared_sample_candidates.csv` | `rank`, `sample_id`, `score`, `criterion`, `selection_threshold_dbm`, 주파수마다 `<f>_RadioUNetM_recall`, `<f>_TransPathNetS_recall`, `<f>_gap_RM_TS` |

### B.6 반복 실험 집계 파일(6.7)

| 파일 | 컬럼 |
|---|---|
| `summary_metrics_by_seed.csv` | 입력 요약 CSV의 모든 컬럼 + `seed`, `source_csv` |
| `summary_metrics_across_seeds.csv` | `threshold_dbm`, `freq_mhz`, `model_tag`, `model_label`, `n_seeds`, `seeds`, `n_samples_mean/min/max`, 지표마다 `<지표>_seed_mean`, `<지표>_seed_std` |
| `paper_table_th<T>.csv` | `freq_mhz`, `model`, `n_seeds`, `seeds`, `precision`, `recall`, `iou`, `f1`, `rmse_dbm`, `mae_dbm`(`"mean +/- std"` 문자열) |
