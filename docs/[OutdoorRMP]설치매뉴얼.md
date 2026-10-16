# Radio Map 예측 기반 음영 영역 검출 설치 매뉴얼

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
  - 1.1 문서 목적
  - 1.2 과제 정보
  - 1.3 관련 문서와 용어
- 2. 제출물 구성
  - 2.1 디렉터리 구조
  - 2.2 실행 방법
  - 2.3 기반 소프트웨어와 신규 코드 범위
- 3. 설치 환경 요구사항
  - 3.1 호스트와 컨테이너
  - 3.2 자원
  - 3.3 인터넷 필요 여부
  - 3.4 설치 구조
  - 3.5 지원 범위
  - 3.6 설치 항목 한눈에 보기
- 4. 설치
  - 4.1 설치 경로 선택
  - 4.2 호스트 점검
  - 4.3 Docker 준비
  - 4.4 압축 해제와 무결성 확인
  - 4.5 경로 A: 제공 환경 이미지로 설치
  - 4.6 경로 B: Dockerfile로 환경 이미지 빌드
- 5. 실행
  - 5.1 명령 형식
  - 5.2 대표 결과 재현
  - 5.3 단일 체크포인트 평가
  - 5.4 모델 학습
  - 5.5 반복 holdout 실험
  - 5.6 결과 파일
- 6. 결과 관리
- 7. 문제 해결
  - 7.1 GPU 메모리 부족
  - 7.2 오류 메시지와 조치
- 8. 코드 변경과 재실행
- 9. 디스크 정리
- 부록 A. 컨테이너 셸 직접 사용
- 부록 B. Windows(WSL 2 + Docker Desktop)에서 설치
  - B.1 사전 환경
  - B.2 파일 복사와 작업 위치
  - B.3 설치와 실행
  - B.4 검증 환경과 결과
- 부록 C. Docker 없이 직접 설치
  - C.1 패키지 설치
  - C.2 실행

## 1. 개요

### 1.1 문서 목적

본 문서는 국방과학연구소가 주관하고 LIG D&A가 수행하는 『5G 특화망의 미래 군통신망 적용 연구 및 시험평가 기술지원』 과제에서 가천대학교 INC 연구실이 위탁 수행한 Radio Map 예측 기반 음영 영역 검출(Radio Map Prediction-based Shadow Region Detection in Sub-7 GHz Environments) 소프트웨어의 설치 매뉴얼이다.

이 문서만으로 다음을 할 수 있도록 작성했다.

- 제공 패키지 `LIG_RMP_SHADOW_v1.0`의 무결성 확인과 실행 환경 구성
- 설치 확인, 대표 결과 재현, 단일 체크포인트 평가, 모델 학습
- 결과 확인과 관리, 코드 변경 후 재실행

Linux 셸과 Docker 기본 명령(docker load, docker compose run)을 다룰 수 있으면 충분하며, 딥러닝 프레임워크(PyTorch) 사용 경험은 필요하지 않다.

### 1.2 과제 정보

| 항목 | 내용 |
|---|---|
| 대상 패키지 | `LIG_RMP_SHADOW_v1.0`(전달 파일 `LIG_RMP_SHADOW_v1.0.tar`). 데이터셋 `LIG_RMP_SHADOW_v1.0_dataset.tar`, 체크포인트 `LIG_RMP_SHADOW_v1.0_checkpoints.tar`는 별도 파일로 전달 |
| 환경 이미지 | `outdoor-rmp:latest`, 이미지 ID `sha256:3771909486423e4ba91306a11a2f20679b450e2b30d090749044404169b3555a`, linux/amd64 |
| 소스 코드 | 패키지의 `ikeee/`, `scripts/`와 Docker 구성 파일. GitHub `https://github.com/james4510/OutdoorRMP`(main)와 같은 내용 |
| 문서 버전 | 1.0 (2026-10-16) |

### 1.3 관련 문서와 용어

| 문서 | 내용 |
|---|---|
| 『설치 매뉴얼』(본 문서) | 설치, 실행, 결과 관리, 문제 해결 |
| 『코드 매뉴얼』 | 데이터셋 로더, 모델, 학습·평가 코드, 분석 스크립트의 구조와 옵션, 출력 지표, 코드 수정과 확장 방법. 부록 A에 CLI 옵션 전체 |

| 용어 | 뜻 |
|---|---|
| 라디오맵(radio map) | 건물 배치와 송신기 위치가 주어졌을 때 지도 위 각 픽셀의 수신 신호 세기(RSS, dBm)를 나타낸 2차원 맵. 본 소프트웨어는 이 맵을 딥러닝 모델로 예측한다 |
| 음영 영역(shadow region) | RSS가 임계값보다 낮은 픽셀의 집합. 건물 픽셀은 제외한다 |
| 고정 임계값 | 음영 판정에 쓰는 RSS 임계값. 대표 결과는 -70, -65, -60 dBm 세 값을 쓰고, 대표 임계값은 -65 dBm이다 |
| 경계 영역 | 정답 RSS가 임계값에서 ±10 dB 안에 있는 픽셀. 임계값 부근의 오차와 miss rate를 따로 평가한다 |
| 모델 | CNN 기반 RadioUNet(S, M)과 Transformer 기반 TransPathNet(S, B)의 4종. 태그는 `radiounet-small`, `radiounet-medium`, `transpathnet-small`, `transpathnet-base` |
| 체크포인트 | 학습된 모델 가중치 파일 `best.pth`. 모델·주파수별로 `model_checkpoints/checkpoints_<모델>_<크기>_<주파수>/`에 둔다 |
| 환경 이미지 | 패키지에 포함된 Docker 이미지(`outdoor-rmp:latest`). PyTorch 2.2.0, CUDA 11.8과 필요한 Python 패키지가 들어 있다 |
| 대표 결과 | 1750, 4750, 5750 MHz에 대해 분석 파이프라인이 만드는 지표·그림 묶음. 패키지의 `results/representative_1750_4750_5750/`에 기준 결과가 있다 |
| 분할 seed | 데이터셋을 학습·검증·테스트로 나눌 때 쓰는 난수 seed. 모든 제공 체크포인트와 결과는 seed 42의 분할을 쓴다 |

## 2. 제출물 구성

### 2.1 디렉터리 구조

아래는 `LIG_RMP_SHADOW_v1.0.tar`를 푼 뒤의 구성이다. 데이터셋과 체크포인트는 별도 tar로 전달되며 4.4에서 이 구조 안에 푼다.

```text
LIG_RMP_SHADOW_v1.0/
├── README.md, README.ko.md      저장소 안내(영문, 한글)
├── SHA256SUMS                   패키지 파일 체크섬
├── MANIFEST.json                버전·구성요소·이미지 식별 정보
├── Dockerfile                   환경 이미지 정의(베이스 pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime)
├── docker-compose.yml           컨테이너 실행 설정(서비스 rmp: GPU, rmp-cpu: CPU)
├── requirements.txt             Docker 없이 설치할 때의 Python 패키지 목록(부록 C)
├── docker/
│   ├── outdoor-rmp_latest.tar.gz        환경 이미지 파일(3.8 GB)과 .sha256
│   ├── requirements.txt                 이미지에 추가로 설치되는 패키지
│   └── setup_env.sh                     실행 환경 설정 파일(.env) 생성
├── docs/                        설치 매뉴얼, 코드 매뉴얼(.docx)
├── ikeee/                       데이터셋 로더(bratlab_dataset.py), 학습·평가(bratlab_train.py), 모델(model/)
├── scripts/                     설치 확인(check_setup.py), 대표 결과 파이프라인, 분석·그림 스크립트
├── results/                     논문 결과(기준 결과). representative_1750_4750_5750/ 등
├── support/
│   └── test_data_40samples.tar.gz       40 샘플 테스트 데이터(106 MB, 동작 확인용)
├── data/                        데이터셋을 푸는 위치(처음에는 비어 있음)
└── model_checkpoints/           체크포인트를 푸는 위치(처음에는 비어 있음)
```

별도 전달 파일은 다음과 같다.

| 파일 | 크기 | 내용 |
|---|---|---|
| `LIG_RMP_SHADOW_v1.0.tar` | 3.9 GB | 위 패키지(환경 이미지 파일 포함) |
| `LIG_RMP_SHADOW_v1.0_dataset.tar` | 13 GB | 데이터셋 2,000 샘플 × 5개 주파수. 풀면 `data/`가 된다 |
| `LIG_RMP_SHADOW_v1.0_checkpoints.tar` | 9.9 GB | 1750, 4750, 5750 MHz의 4개 모델 체크포인트 13개(4750 MHz RadioUNet-S의 v2 포함). 풀면 `model_checkpoints/`가 된다 |
| `*.sha256` | – | 각 tar의 체크섬 |

- 과제 코드는 `ikeee/`(데이터셋 로더, 모델 4종, 학습·평가)와 `scripts/`(설치 확인, 파이프라인, 분석·그림, 반복 실험 스크립트 15개)에 있다. 구성은 『코드 매뉴얼』 2장에 있다.
- 실행하면 다음 위치에 새 파일이 생긴다. 모두 패키지 안이며 다른 위치에는 쓰지 않는다.
  - 실행 환경 설정: `.env`(4.5.2)
  - 대표 결과 재현: `results_1750_4750_5750_regenerated/`(5.2)
  - 단일 체크포인트 평가: `--result_dir`로 지정한 폴더. 본 문서는 `runs/` 아래를 쓴다(5.3)
  - 학습: `--save_dir`로 지정한 폴더. 본 문서는 `model_checkpoints/checkpoints_<모델>_<크기>_<주파수>/`를 쓴다(5.4)
  - 반복 holdout 실험: `results_multi_seed_holdout/`(5.5)

### 2.2 실행 방법

| 방법 | 용도 | 본 문서 |
|---|---|---|
| 설치 확인 `scripts/check_setup.py` | 패키지, GPU, 코드, 데이터셋, 체크포인트 점검 | 4.5.3 |
| 대표 결과 파이프라인 `scripts/run_1750_4750_5750_shared_pipeline.sh` (권장) | 제공 체크포인트로 1750/4750/5750 MHz의 지표·CDF·정성 그림을 한 번에 생성 | 5.2 |
| 단일 체크포인트 평가 `python -m ikeee.bratlab_train --mode eval` | 체크포인트 하나의 RMSE·MAE·bias·SSIM과 예측 그림 | 5.3 |
| 모델 학습 `python -m ikeee.bratlab_train --mode train` | 주파수·모델을 골라 체크포인트 생성 | 5.4 |
| 반복 holdout 실험 `scripts/run_multi_seed_holdout_experiments.sh` | 분할 seed를 바꿔 학습·평가·집계 | 5.5 |

모든 명령은 패키지 루트에서 `docker compose run --rm rmp <명령>` 형태로 컨테이너 안에서 실행한다(5.1).

### 2.3 기반 소프트웨어와 신규 코드 범위

| 구분 | 버전·범위 | 비고 |
|---|---|---|
| PyTorch | 2.2.0 (CUDA 11.8, cuDNN 8), torchvision 0.17.0 | 베이스 이미지 `pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime`에 포함 |
| 추가 Python 패키지 | numpy 1.26, scipy 1.15.3, matplotlib 3.10.0, pillow, pytorch-msssim 1.0.0 | `docker/requirements.txt`로 이미지에 설치 |
| 과제 코드 | `ikeee/` 모듈 6개(데이터셋 로더, 학습·평가, 모델 4개), `scripts/` 15개(파이썬 12개, 셸 3개) | 모두 본 과제에서 작성 |
| 참조 구현 | RadioUNet(Levie et al., IEEE TWC 2021), TransPathNet(LiXin97/TransPathNet), TransNeXt(Apache 2.0), EMCAD(BSD 3-Clause) | 논문·공개 코드의 구조를 2채널 입력에 맞게 PyTorch로 재구현. 외부 코드를 그대로 포함하지 않음 |

## 3. 설치 환경 요구사항

### 3.1 호스트와 컨테이너

| 구분 | 호스트 | 컨테이너(환경 이미지) |
|---|---|---|
| OS·아키텍처 | Linux x86_64(Ubuntu 20.04/22.04). Windows 11은 WSL 2 + Docker Desktop(부록 B) | Ubuntu 22.04, linux/amd64 |
| GPU | NVIDIA GPU와 드라이버(CUDA 11.8 지원, 드라이버 520 이상 권장) | CUDA 11.8, cuDNN 8 |
| 소프트웨어 | Docker Engine, Docker Compose v2, NVIDIA Container Toolkit, bash, tar, coreutils(sha256sum) | Python 3.10, PyTorch 2.2.0, torchvision 0.17.0, numpy, scipy, matplotlib, pillow, pytorch-msssim |
| 파일 연결 | – | 패키지 루트 → `/workspace`, `data/` → `/workspace/data`(읽기 전용), `model_checkpoints/` → `/workspace/model_checkpoints` |
| 실행 사용자 | 일반 사용자(docker 그룹) | 호스트 사용자와 같은 UID/GID(`.env`의 `HOST_UID`, `HOST_GID`) |

`docker-compose.yml`이 패키지 루트를 컨테이너의 `/workspace`에 연결하므로, 컨테이너가 만드는 결과 파일은 모두 호스트의 패키지 안에 남고 소유자는 호스트 사용자가 된다. 데이터셋은 읽기 전용으로 연결되어 컨테이너가 바꿀 수 없다.

### 3.2 자원

| 항목 | 권장 | 최소 |
|---|---|---|
| GPU 메모리 | 11 GB 이상(학습 batch size 8) | 4 GB(평가·대표 결과 재현. 학습은 batch size를 줄이고 AMP 사용) |
| CPU | 8코어 | 4코어 |
| 메모리 | 16 GB | 8 GB |
| 디스크 여유 | 50 GB | 패키지 3.9 GB + 불러온 이미지 약 12.5 GB + 데이터셋 13 GB + 체크포인트 10 GB + 결과 |

- 대표 결과 재현은 모델 4개 × 주파수 3개의 테스트 400 샘플 추론을 세 단계(지표, 경계 영역, 정성 그림)에서 반복하므로 GPU 성능에 따라 수십 분에서 수 시간이 걸린다.
- 학습은 100 epoch 기준 모델과 GPU에 따라 수 시간 이상 걸린다. 동작 확인은 40 샘플 테스트 데이터로 1 epoch만 학습한다(4.5.5).

### 3.3 인터넷 필요 여부

| 단계 | 인터넷 |
|---|---|
| 경로 A(환경 이미지 로드 → 설치 확인 → 실행 → 결과 확인) | 불필요(패키지만으로 전 과정 수행 가능) |
| 경로 B(Dockerfile로 환경 이미지 새로 빌드) | 필요(Docker Hub, PyPI) |
| Docker Engine, NVIDIA Container Toolkit 설치(4.3) | 필요(Docker·NVIDIA apt 저장소). 인터넷이 없는 호스트는 다른 PC에서 받은 .deb 파일로 설치(4.3.5) |
| 부록 C(Docker 없이 직접 설치) | 필요(PyPI) |

### 3.4 설치 구조

환경 이미지 파일 → (docker load, 1회) → 이미지 `outdoor-rmp:latest` → (docker compose run --rm rmp <명령>) → 컨테이너(호스트와 같은 UID/GID; `/workspace` = 패키지 루트, `/workspace/data` = 데이터셋, `/workspace/model_checkpoints` = 체크포인트) → 결과는 패키지 안에 남는다.

- 환경 이미지 파일을 docker load로 한 번 등록한다.
- 이후 `docker compose run --rm rmp`가 명령마다 컨테이너를 만들고 패키지를 연결하며, 끝나면 컨테이너를 지운다(`--rm`).
- 이미지에도 소스 사본이 들어 있지만 실행할 때 호스트의 패키지 루트가 `/workspace`를 덮어 연결하므로, 코드를 고치면 재빌드 없이 바로 반영된다(8장).
- `docker/setup_env.sh`가 만드는 `.env`에 사용자 UID/GID와 데이터셋·체크포인트 경로가 기록되고, `docker-compose.yml`이 이를 읽는다.

### 3.5 지원 범위

본 패키지는 다음 환경에서 설치 확인, 1 epoch 학습, 단일 체크포인트 평가, 대표 결과 재현까지 검증했다(부록 B.4).

- Windows 11 25H2 + WSL 2(Ubuntu 26.04) + Docker Desktop(Docker 29.4.3, Compose 5.1.3) + NVIDIA GeForce GTX 1050(드라이버 560.94)

논문 결과(`results/`)는 Ubuntu 20.04 + RTX 2080 Ti(드라이버 570.172.08) 호스트에서 같은 버전의 PyTorch 2.2.0 + CUDA 11.8로 생성했다. Linux x86_64 호스트의 Docker Engine 환경은 같은 이미지와 compose 파일을 쓰므로 본 문서의 절차를 그대로 적용한다. 그 외 환경은 지원 범위 밖이다.

### 3.6 설치 항목 한눈에 보기

PyTorch와 CUDA 런타임은 따로 설치하지 않는다. 환경 이미지에 들어 있으며, 호스트에는 NVIDIA 드라이버와 Docker만 있으면 된다. 아래 표는 설치 경로별로 각 항목을 어떻게 준비하는지와 해당 절을 정리한 것이다.

| 항목 | 경로 A(제공 환경 이미지) | 경로 B(Dockerfile) | 부록 B(Windows WSL 2) | 부록 C(직접 설치) |
|---|---|---|---|---|
| Docker | 호스트에 Docker Engine 설치(4.3) | 경로 A와 같음 | Docker Desktop 설치(B.1) | 필요 없음 |
| NVIDIA Container Toolkit | 호스트에 설치(4.3.4) | 경로 A와 같음 | Docker Desktop에 포함 | 필요 없음 |
| 환경 이미지 | 패키지의 이미지 파일을 로드(4.5.1) | `docker compose build`(4.6) | 경로 A 또는 B와 같음 | 사용하지 않음 |
| PyTorch·CUDA 런타임 | 환경 이미지에 포함 | Dockerfile이 베이스 이미지에서 가져옴 | 경로 A와 같음 | pip로 설치(C.1) |
| 과제 코드 | 패키지에 포함 | 경로 A와 같음 | 경로 A와 같음 | 경로 A와 같음 |
| 데이터셋·체크포인트 | 별도 tar를 패키지 안에 품(4.4) | 경로 A와 같음 | 경로 A와 같음(WSL 홈 아래) | 경로 A와 같음 |

## 4. 설치

### 4.1 설치 경로 선택

| 경로 | 언제 | 절차 |
|---|---|---|
| A. 제공 환경 이미지 사용(권장) | 연구에 사용한 것과 같은 PyTorch·CUDA 구성으로 실행. 인터넷 불필요 | 4.2 → 4.3 → 4.4 → 4.5 → 5장 |
| B. Dockerfile로 환경 이미지 빌드 | 제공 이미지 파일을 쓸 수 없을 때 | 4.2 → 4.3 → 4.4 → 4.6 → 4.5.2 이후 |
| Windows PC | Windows 11에서 WSL 2 + Docker Desktop으로 실행 | 부록 B → 4.4 → 4.5 → 5장 |
| Docker 없이 직접 설치 | Docker를 쓸 수 없는 Linux 호스트 | 4.2 → 4.4 → 부록 C |

### 4.2 호스트 점검

**실행 위치**: 호스트 · 작업 디렉터리: 임의

```bash
uname -m
. /etc/os-release && echo "$PRETTY_NAME"
nproc
free -g | awk '/^Mem:/{print $2" GiB"}'
df -h ~
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

정상 출력 예(마지막 명령)

```text
name, driver_version, memory.total [MiB]
NVIDIA GeForce RTX 2080 Ti, 570.172.08, 11264 MiB
```

- 정상 기준: x86_64, Ubuntu 20.04 또는 22.04, 3.2절의 CPU·메모리·디스크, NVIDIA 드라이버 520 이상
- `nvidia-smi`가 없으면 NVIDIA 드라이버를 먼저 설치한다(Ubuntu: `sudo ubuntu-drivers install` 또는 NVIDIA 공식 드라이버). 드라이버가 없으면 GPU 서비스 `rmp`를 쓸 수 없고 `rmp-cpu`만 쓸 수 있다.
- 없는 도구 설치(Ubuntu): `sudo apt update && sudo apt install -y tar coreutils`

### 4.3 Docker 준비

#### 4.3.1 설치 여부 확인

**실행 위치**: 호스트 · 작업 디렉터리: 임의

```bash
docker version
docker compose version
docker info --format '{{.OSType}}/{{.Architecture}} {{.ServerVersion}}'
```

정상 출력 예(`docker version`, `docker compose version`)

```text
Client: Docker Engine - Community
 Version:           28.1.1
 ...
Server: Docker Engine - Community
 Engine:
  Version:          28.1.1
 ...
Docker Compose version v2.35.1
```

- Client와 Server가 모두 표시되고 Compose 버전이 v2 이상이면 Docker가 준비된 것이다. 세 번째 명령은 `linux/x86_64 <버전>`을 출력한다.
- Server 부분 대신 `permission denied`가 나오면 4.3.3을, `command not found`이면 4.3.2(인터넷이 없으면 4.3.5)를 수행한다.
- Docker가 준비되어 있어도 GPU를 쓰려면 4.3.4의 NVIDIA Container Toolkit이 필요하다.

#### 4.3.2 Docker Engine 설치(인터넷 연결 호스트)

Docker 공식 문서의 Ubuntu apt 저장소 설치 절차이다(출처: `https://docs.docker.com/engine/install/ubuntu/`). 관리자 권한이 필요하다.

**실행 위치**: 호스트 · 작업 디렉터리: 임의

```bash
sudo apt remove $(dpkg --get-selections docker.io docker-compose docker-compose-v2 docker-doc docker-buildx podman-docker containerd runc | cut -f1)
# Add Docker's official GPG key:
sudo apt update
sudo apt install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
# Add the repository to Apt sources:
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo docker version
```

- 첫 명령은 Ubuntu가 제공하는 비공식 Docker 패키지 등 충돌 패키지를 지운다. 설치된 것이 없다는 메시지가 나와도 된다.
- 마지막 `sudo docker version`에서 Client와 Server가 모두 표시되면 설치된 것이다(4.3.1의 출력 예와 같은 형식).

#### 4.3.3 일반 사용자로 Docker 실행

공식 문서의 설치 후 절차이다(출처: `https://docs.docker.com/engine/install/linux-postinstall/`).

**실행 위치**: 호스트 · 작업 디렉터리: 임의

```bash
sudo groupadd docker
sudo usermod -aG docker $USER
```

- 로그아웃한 뒤 다시 로그인해야 그룹 변경이 적용된다. 현재 셸에만 바로 적용하려면 `newgrp docker`를 실행한다.
- 패키지 설치 과정에서 docker 그룹이 이미 만들어졌으면 groupadd가 그룹이 있다고 알린다. 무시해도 된다.
- 다시 로그인한 뒤 sudo 없이 `docker version`을 실행해 Server가 표시되는지 확인한다.
- docker 그룹은 root와 같은 권한을 가지므로 기관 보안 정책을 확인한다.

#### 4.3.4 NVIDIA Container Toolkit 설치

컨테이너에서 GPU를 쓰기 위한 도구이다. NVIDIA 공식 문서의 apt 설치 절차이다(출처: `https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html`).

**실행 위치**: 호스트 · 작업 디렉터리: 임의

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker info --format '{{json .Runtimes}}'
```

정상 출력 예(마지막 명령)

```text
{"io.containerd.runc.v2":{"path":"runc"},"nvidia":{"path":"nvidia-container-runtime"},"runc":{"path":"runc"}}
```

- 출력에 `nvidia` 런타임이 보이면 설정된 것이다. 컨테이너에서 GPU가 실제로 보이는지는 환경 이미지를 로드한 뒤 4.5.4에서 확인한다.
- Docker Desktop(Windows WSL 2)에는 이 도구가 포함되어 있어 설치하지 않는다(부록 B).

#### 4.3.5 인터넷이 없는 호스트

Docker 공식 문서의 패키지 파일 설치 방법을 쓴다(출처: `https://docs.docker.com/engine/install/ubuntu/`의 "Install from a package").

1. 인터넷이 되는 PC에서 `https://download.docker.com/linux/ubuntu/dists/<코드명>/pool/stable/amd64/`에 접속해 .deb 파일 5개(containerd.io, docker-ce, docker-ce-cli, docker-buildx-plugin, docker-compose-plugin)를 받는다. 코드명은 대상 호스트에서 `. /etc/os-release && echo "$VERSION_CODENAME"`으로 확인한다(Ubuntu 22.04는 jammy, 20.04는 focal).
2. NVIDIA Container Toolkit의 .deb 파일(libnvidia-container1, libnvidia-container-tools, nvidia-container-toolkit-base, nvidia-container-toolkit)도 `https://nvidia.github.io/libnvidia-container/stable/deb/amd64/`에서 받는다.
3. 받은 파일을 대상 호스트로 옮겨 설치한다.

**실행 위치**: 호스트 · 작업 디렉터리: .deb 파일을 둔 디렉터리

```bash
sudo dpkg -i ./containerd.io_<version>_<arch>.deb \
  ./docker-ce_<version>_<arch>.deb \
  ./docker-ce-cli_<version>_<arch>.deb \
  ./docker-buildx-plugin_<version>_<arch>.deb \
  ./docker-compose-plugin_<version>_<arch>.deb
sudo dpkg -i ./libnvidia-container1_*.deb ./libnvidia-container-tools_*.deb \
  ./nvidia-container-toolkit-base_*.deb ./nvidia-container-toolkit_*.deb
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo docker version
```

- dpkg가 의존 패키지가 없다고 알리면, 그 Ubuntu 패키지도 같은 방법으로 받아 함께 설치한다.
- 이어서 4.3.3으로 일반 사용자 권한을 설정한다.
- Docker 설치가 끝나면 경로 A(4.4 → 4.5 → 5장)는 인터넷 없이 진행할 수 있다.

### 4.4 압축 해제와 무결성 확인

전달받은 `LIG_RMP_SHADOW_v1.0.tar`, `LIG_RMP_SHADOW_v1.0_dataset.tar`, `LIG_RMP_SHADOW_v1.0_checkpoints.tar`와 각 `.sha256`를 Linux 호스트의 작업 디렉터리로 복사한 뒤, 체크섬을 확인하고 푼다. 데이터셋과 체크포인트 tar는 패키지 루트에서 풀면 `data/`, `model_checkpoints/`에 들어간다.

**실행 위치**: 호스트 · 작업 디렉터리: 아카이브가 있는 디렉터리

```bash
sha256sum -c LIG_RMP_SHADOW_v1.0.tar.sha256 LIG_RMP_SHADOW_v1.0_dataset.tar.sha256 LIG_RMP_SHADOW_v1.0_checkpoints.tar.sha256
mkdir -p ~/rmp-work
tar -xf LIG_RMP_SHADOW_v1.0.tar -C ~/rmp-work
tar -xf LIG_RMP_SHADOW_v1.0_dataset.tar -C ~/rmp-work/LIG_RMP_SHADOW_v1.0
tar -xf LIG_RMP_SHADOW_v1.0_checkpoints.tar -C ~/rmp-work/LIG_RMP_SHADOW_v1.0
cd ~/rmp-work/LIG_RMP_SHADOW_v1.0
sha256sum --quiet -c SHA256SUMS
ls data model_checkpoints
```

정상 출력 예

```text
LIG_RMP_SHADOW_v1.0.tar: OK
LIG_RMP_SHADOW_v1.0_dataset.tar: OK
LIG_RMP_SHADOW_v1.0_checkpoints.tar: OK
data:
buildings_position  receivedpower_1750MHz_mat  receivedpower_2750MHz_mat  receivedpower_3750MHz_mat  receivedpower_4750MHz_mat  receivedpower_5750MHz_mat  stations_position.txt

model_checkpoints:
checkpoints_radiounet_medium_1750   checkpoints_radiounet_small_4750     checkpoints_transpathnet_base_4750
checkpoints_radiounet_medium_4750   checkpoints_radiounet_small_4750_v2  checkpoints_transpathnet_base_5750
checkpoints_radiounet_medium_5750   checkpoints_radiounet_small_5750     checkpoints_transpathnet_small_1750
checkpoints_radiounet_small_1750    checkpoints_transpathnet_base_1750   checkpoints_transpathnet_small_4750
checkpoints_transpathnet_small_5750
```

- `sha256sum --quiet -c SHA256SUMS`는 패키지의 모든 파일이 일치하면 아무것도 출력하지 않는다. 한국어 로캘에서는 OK 대신 성공이 출력된다.
- `SHA256SUMS`는 패키지 파일만 다루며 `data/`, `model_checkpoints/` 안의 파일은 각 tar의 `.sha256`로 확인한다.
- 데이터셋 13 GB를 푸는 데 수 분이 걸린다. 디스크가 다른 위치에 있으면 거기에 풀고 4.5.2에서 경로를 지정한다.
- 무결성 확인은 실행 전에 한다. 실행 후에는 결과 파일이 추가될 뿐 `SHA256SUMS`에 적힌 파일은 바뀌지 않는다.

### 4.5 경로 A: 제공 환경 이미지로 설치

#### 4.5.1 환경 이미지 로드

**실행 위치**: 호스트 · 작업 디렉터리: 패키지 루트

```bash
cd docker && sha256sum -c outdoor-rmp_latest.tar.gz.sha256 && cd ..
docker load -i docker/outdoor-rmp_latest.tar.gz
docker image inspect outdoor-rmp:latest --format '{{.Id}} {{.Os}}/{{.Architecture}}'
```

정상 출력 예

```text
outdoor-rmp_latest.tar.gz: OK
Loaded image: outdoor-rmp:latest
sha256:3771909486423e4ba91306a11a2f20679b450e2b30d090749044404169b3555a linux/amd64
```

- 인터넷 없이 수행된다(이미지 파일 3.8 GB, 로드 후 약 12.5 GB). 로드에 수 분이 걸린다.
- 이미지 이름은 `outdoor-rmp:latest`이다. 같은 이름의 다른 이미지가 이미 있으면 덮어쓰므로 이미지 ID로 확인한다.

#### 4.5.2 실행 환경 설정 파일(.env) 생성

**실행 위치**: 호스트 · 작업 디렉터리: 패키지 루트

```bash
bash docker/setup_env.sh
cat .env
```

정상 출력 예

```text
Created /home/user/rmp-work/LIG_RMP_SHADOW_v1.0/.env
HOST_UID=1000
HOST_GID=1000
DATA_DIR=/home/user/rmp-work/LIG_RMP_SHADOW_v1.0/data
CKPT_DIR=/home/user/rmp-work/LIG_RMP_SHADOW_v1.0/model_checkpoints
```

- `HOST_UID`, `HOST_GID`는 컨테이너가 만든 결과 파일의 소유자를 현재 사용자로 맞추는 값이다. 다른 사용자 계정으로 실행할 때는 다시 생성한다.
- 데이터셋과 체크포인트가 다른 위치에 있으면 `bash docker/setup_env.sh /path/to/data /path/to/model_checkpoints`로 지정한다. 데이터셋 폴더가 없으면 `[ERROR] Dataset folder not found`로 끝난다.

#### 4.5.3 설치 확인

**실행 위치**: 호스트 · 작업 디렉터리: 패키지 루트

```bash
docker compose run --rm rmp python scripts/check_setup.py
```

정상 출력 예

```text
[1/5] Python packages
  OK       torch           2.2.0
  OK       torchvision     0.17.0
  OK       numpy           1.26.3
  OK       scipy           1.15.3
  OK       matplotlib      3.10.0
  OK       PIL             10.0.1
  OK       pytorch_msssim  1.0.0
[2/5] GPU
  OK       cuda:0  NVIDIA GeForce RTX 2080 Ti
[3/5] Project code (ikeee)
  OK       ikeee.bratlab_dataset, ikeee.model
[4/5] Dataset: /workspace/data
  OK       stations_position.txt (2000 samples)
  OK       buildings_position (2000 files)
  OK       receivedpower_1750MHz_mat (2000 files)
  ...
  OK       receivedpower_5750MHz_mat (2000 files)
[5/5] Checkpoints: /workspace/model_checkpoints
  model                    1750   2750   3750   4750   5750
  radiounet_small             O      -      -      O      O
  radiounet_medium            O      -      -      O      O
  transpathnet_small          O      -      -      O      O
  transpathnet_base           O      -      -      O      O
  (checkpoints_radiounet_small_4750_v2 found: used for RadioUNet-S at 4750 MHz)
  O: best.pth found, -: not found

Result: OK
```

- `Result: OK`는 패키지·코드·데이터셋이 준비되었다는 뜻이다. 체크포인트는 표로만 보여 주며 판정에 들어가지 않는다. 제공 체크포인트 기준으로는 1750, 4750, 5750 열만 `O`이다.
- 처음 실행할 때 `groups: cannot find name for group ID 1000` 같은 메시지가 나올 수 있다. 호스트 UID로 실행하면서 나오는 것으로 문제없다.
- `[2/5] GPU`에 `WARN CUDA is not available`이 나오면 4.3.4를 확인한다. `could not select device driver`로 실행 자체가 실패하면 NVIDIA Container Toolkit이 설정되지 않은 것이다(7.2).
- MISSING 또는 FAIL 항목이 있으면 해당 경로와 파일을 확인한다. 패키지가 하나라도 MISSING이면 `Result: FAIL`이다.

#### 4.5.4 동작 확인

제공 체크포인트 하나로 테스트 분할 전체(400 샘플)를 평가해 GPU 추론과 결과 저장이 되는지 확인한다.

**실행 위치**: 호스트 · 작업 디렉터리: 패키지 루트

```bash
docker compose run --rm rmp python -m ikeee.bratlab_train --mode eval \
    --root data --freq 5750 --model radiounet --size small \
    --ckpt model_checkpoints/checkpoints_radiounet_small_5750/best.pth \
    --result_dir runs/eval_check --n_vis 3
ls runs/eval_check
```

정상 출력 예

```text
[BRATLabDataset] 2000개 유효 샘플 | freq=5750MHz | img_size=256 | augment=False
...
[DataLoader-Fixed] Train:1400 | Val:200 | Test:400
[DataLoader-Fixed] augment: train=True, val=False, test=False
[Eval] radiounet-small | Seed: 42 | 체크포인트: model_checkpoints/checkpoints_radiounet_small_5750/best.pth (epoch 98)
  저장: runs/eval_check/result_001_sample990.png
  저장: runs/eval_check/result_002_sample1627.png
  저장: runs/eval_check/result_003_sample1998.png

=============================================
  Model      : radiounet-small
  Freq       : 5750 MHz
  Test RMSE  : 2.1957 dBm
  Test MAE   : 1.5755 dBm
  Test Bias  : -0.0673 dB
  Test SSIM  : 0.9422
=============================================
result_001_sample990.png  result_002_sample1627.png  result_003_sample1998.png
```

- 마지막 표가 출력되고 `runs/eval_check/`에 PNG 3개가 생기면 정상이다. Test RMSE·MAE는 `results/representative_1750_4750_5750/metrics/summary_metrics_fixed_threshold.csv`의 5750 MHz RadioUNet-S 행(`rmse_dbm_mean` 2.196, `mae_dbm_mean` 1.575)과 소수점 셋째 자리까지 같아야 한다.
- `UserWarning: Applied workaround for CuDNN issue, install nvrtc.so`는 PyTorch 2.2의 경고로 결과에 영향이 없다.
- `CUDA out of memory`가 나오면 7.1을 본다. GPU 없이 확인하려면 `rmp`를 `rmp-cpu`로 바꾼다(수 분 소요).

#### 4.5.5 테스트 데이터로 동작 확인(선택)

전체 데이터셋을 받기 전이거나 GPU 메모리가 작은 PC에서 학습까지 확인하려면, `support/`의 40 샘플 테스트 데이터로 1 epoch 학습과 평가를 수행한다. 이 절은 `data/`에 전체 데이터셋을 풀기 전에 수행하거나, 테스트 데이터를 다른 폴더에 풀어 경로를 지정한다.

**실행 위치**: 호스트 · 작업 디렉터리: 패키지 루트

```bash
mkdir -p ../rmp-test && tar -xzf support/test_data_40samples.tar.gz -C ../rmp-test     # ../rmp-test/data 생성
bash docker/setup_env.sh "$(cd ../rmp-test/data && pwd)" model_checkpoints
docker compose run --rm rmp python scripts/check_setup.py
docker compose run --rm rmp python -m ikeee.bratlab_train --mode train \
    --root data --freq 4750 --model radiounet --size small \
    --batch_size 4 --amp --epochs 1 --num_workers 2 \
    --save_dir model_checkpoints/checkpoints_test_radiounet_small_4750
docker compose run --rm rmp python -m ikeee.bratlab_train --mode eval \
    --root data --freq 4750 --model radiounet --size small \
    --ckpt model_checkpoints/checkpoints_test_radiounet_small_4750/best.pth \
    --result_dir runs/eval_test
bash docker/setup_env.sh                       # 전체 데이터셋 경로로 되돌림
```

정상 출력 예(학습)

```text
[DataLoader-Fixed] Train:28 | Val:4 | Test:8
[Train] Device: cuda | Model: radiounet-small | Seed: 42
[Model] radiounet-small | Params: 34.92M
[Train] Batch size: 4 | Accum steps: 1 | AMP: True
[001/1] Train:0.xxxx Val:0.xxxx | RMSE:0.xxxx MAE:0.xxxx SSIM:0.xxxx | 9.8s
  ✅ Best saved → model_checkpoints/checkpoints_test_radiounet_small_4750/best.pth
  완료 표시 저장 → model_checkpoints/checkpoints_test_radiounet_small_4750/training_complete.txt
[Train] radiounet-small 완료!
```

- 설치 확인에서는 데이터셋 폴더마다 `(40 files)`로 표시된다. 학습 로그의 손실·지표 값(`x`로 표시)은 데이터와 GPU에 따라 다르다. 평가 끝의 Test RMSE 등은 1 epoch 학습 결과라 큰 값(수십 dBm)이 나오며, 오류 없이 끝나는지만 본다.
- 학습 로그의 RMSE·MAE는 정규화된 [0, 1] 단위이고, 평가의 RMSE·MAE는 dBm 단위이다(『코드 매뉴얼』 8.2).
- 확인이 끝나면 마지막 명령으로 `.env`를 전체 데이터셋 경로로 되돌린다. 테스트용 체크포인트(`checkpoints_test_*`)와 `runs/`는 지워도 된다(9장).

### 4.6 경로 B: Dockerfile로 환경 이미지 빌드

제공 이미지 파일 대신 패키지의 `Dockerfile`로 같은 구성의 이미지를 새로 만든다.

**실행 위치**: 호스트 · 작업 디렉터리: 패키지 루트

```bash
docker compose build
docker images outdoor-rmp
```

정상 출력 예

```text
REPOSITORY    TAG       IMAGE ID       CREATED          SIZE
outdoor-rmp   latest    xxxxxxxxxxxx   10 seconds ago   12.5GB
```

- 베이스 이미지 `pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime`(약 4 GB)을 Docker Hub에서 받고, `docker/requirements.txt`의 패키지를 PyPI에서 설치한다. 인터넷이 필요하며 10분 이상 걸릴 수 있다.
- 만들어진 이미지 이름은 제공 이미지와 같은 `outdoor-rmp:latest`이므로 이후 절차(4.5.2부터)는 경로 A와 같다. 이미지 ID는 제공 이미지와 다르다.
- 베이스 이미지는 태그로 고정되어 있고 추가 패키지는 버전을 고정했다(numpy<2, scipy 1.15.3, matplotlib 3.10.0, pytorch-msssim 1.0.0). 빌드 시점의 Docker Hub 이미지가 갱신되면 세부 버전이 달라질 수 있다.

## 5. 실행

### 5.1 명령 형식

| 명령 | 동작 | GPU |
|---|---|---|
| `docker compose run --rm rmp <명령>` | GPU 컨테이너에서 명령 실행. 끝나면 컨테이너 삭제 | 필요 |
| `docker compose run --rm rmp-cpu <명령>` | 같은 이미지를 GPU 없이 실행(평가, 그림 생성용. 학습은 비권장) | 불필요 |
| `docker compose run --rm -e VAR=값 rmp <명령>` | 환경 변수를 넘겨 실행(파이프라인의 `ROOT`, `CKPT_DIR`, `OUT_DIR`, 반복 실험의 `SEEDS` 등) | – |
| `docker compose run --rm rmp bash` | 컨테이너 셸로 들어감(부록 A) | – |

- 컨테이너 안의 작업 디렉터리는 `/workspace`(= 패키지 루트)이므로 명령의 경로는 패키지 루트 기준 상대 경로로 쓴다.
- 파이썬 모듈은 `python -m ikeee.bratlab_train`처럼 실행하고, 스크립트는 `python scripts/<파일>` 또는 `bash scripts/<파일>`로 실행한다.
- 긴 작업은 `2>&1 | tee <로그 파일>`을 붙여 로그를 남긴다. 로그 파일도 패키지 안에 생긴다.

### 5.2 대표 결과 재현

#### 5.2.1 실행

**실행 위치**: 호스트 · 작업 디렉터리: 패키지 루트

```bash
docker compose run --rm rmp bash scripts/run_1750_4750_5750_shared_pipeline.sh 2>&1 | tee pipeline.log
```

정상 출력 예(발췌)

```text
============================================================
[1/4] Fixed-threshold evaluation metrics
============================================================
[Import] Using get_dataloaders
Device: cuda | Split seed: 42

================================================================================
Frequency: 1750 MHz
================================================================================
  Loaded radiounet-small     | epoch=97 | /workspace/model_checkpoints/checkpoints_radiounet_small_1750/best.pth
  ...
[DataLoader-Fixed] Train:1400 | Val:200 | Test:400
  50/400 batches done
  ...
Fixed threshold = -65.0 dBm | sample-wise mean
  Freq   Model                  Prec    Recall  IoU     F1
  ...
Saved: /workspace/results_1750_4750_5750_regenerated/metrics/summary_metrics_fixed_threshold.csv
...
[2/4] Sample-wise Miss Rate CDF
...
[3/4] Boundary-only CDFs
...
[4/4] Shared-sample qualitative figures
...
[Sample Selection] selected sample_id=770 | score=0.1194 | criterion=rm_minus_ts_recall_gap
...
Done.
Results saved in: /workspace/results_1750_4750_5750_regenerated
Main output folders:
  - Metrics:        /workspace/results_1750_4750_5750_regenerated/metrics
  - Miss CDF:       /workspace/results_1750_4750_5750_regenerated/miss_cdf
  - Boundary CDF:   /workspace/results_1750_4750_5750_regenerated/boundary_cdf
  - Qualitative:    /workspace/results_1750_4750_5750_regenerated/qualitative
```

- 네 단계를 순서대로 수행한다: (1) 고정 임계값 지표와 heatmap, (2) 샘플별 miss rate CDF, (3) 경계 영역 오차·miss rate CDF, (4) 세 주파수 공통 샘플의 정성 그림. 단계와 지표의 정의는 『코드 매뉴얼』 6장에 있다.
- (1), (3), (4)는 모델 추론을 수행하고 (2)는 (1)의 CSV만 읽는다. 전체 소요 시간은 GPU에 따라 수십 분에서 수 시간이다.
- 출력 폴더는 `results_1750_4750_5750_regenerated/`이다. 다른 위치에 쓰려면 `-e OUT_DIR=/workspace/<폴더>`를 붙인다. 이미 있는 폴더에 다시 실행하면 같은 이름의 파일을 덮어쓴다.
- 데이터셋과 체크포인트 경로는 `.env`로 연결된 `/workspace/data`, `/workspace/model_checkpoints`를 쓰므로 따로 지정하지 않는다.

#### 5.2.2 결과 확인

**실행 위치**: 호스트 · 작업 디렉터리: 패키지 루트

```bash
ls results_1750_4750_5750_regenerated/*
diff <(cut -d, -f2- results_1750_4750_5750_regenerated/metrics/summary_metrics_fixed_threshold.csv) \
     results/representative_1750_4750_5750/metrics/summary_metrics_fixed_threshold.csv | head
cat results_1750_4750_5750_regenerated/qualitative/selected_shared_sample.csv | cut -d, -f1-5
```

정상 출력 예

```text
results_1750_4750_5750_regenerated/boundary_cdf:
README_boundary_cdf.txt  boundary_cdf_summary.csv  fig_boundary_combined_cdf_x10_th-60_bw10.png ...
results_1750_4750_5750_regenerated/metrics:
README_fixed_threshold_metrics.txt  heatmap_fixed_threshold_th-60.png ... per_sample_metrics_fixed_threshold.csv  summary_metrics_fixed_threshold.csv ...
results_1750_4750_5750_regenerated/miss_cdf:
README_v3.txt  cdf_summary_v3.csv  fig_cdf_missrate_main_th-60.png ...
results_1750_4750_5750_regenerated/qualitative:
error_map_4models_1750MHz_sample770_th-65.png ... selected_shared_sample.csv  shadow_detection_allfreqs_sample770_th-65.png
(diff 출력: RadioUNet 행은 없음, TransPathNet 행은 소수점 6째 자리 이하 차이)
sample_id,freq_mhz,score,criterion,selection_threshold_dbm
770,1750,0.1194072514608856,rm_minus_ts_recall_gap,-65.0
770,4750,...
770,5750,...
```

- 네 폴더에 `results/representative_1750_4750_5750/`와 같은 이름의 CSV·PNG가 생기면 성공이다. 재현 CSV에는 `seed` 열이 하나 더 있으므로 비교할 때 첫 열을 제외한다.
- 같은 체크포인트와 분할(seed 42)을 쓰므로 RadioUNet 행은 기준 결과와 완전히 같고, TransPathNet 행은 GPU 종류에 따라 소수점 6째 자리 이하에서 차이가 날 수 있다(『코드 매뉴얼』 8.3).
- 정성 그림의 공통 샘플은 지표 CSV에서 자동 선택되며 기준 결과와 같은 770번이 선택되어야 한다.

### 5.3 단일 체크포인트 평가

**실행 위치**: 호스트 · 작업 디렉터리: 패키지 루트

```bash
docker compose run --rm rmp python -m ikeee.bratlab_train --mode eval \
    --root data --freq 5750 --model transpathnet --size base \
    --ckpt model_checkpoints/checkpoints_transpathnet_base_5750/best.pth \
    --result_dir runs/eval_transpathnet_base_5750
```

- 테스트 분할(seed 42)의 샘플마다 건물을 제외한 픽셀에서 RMSE, MAE, bias(예측 − 정답, dB)를 계산하고 SSIM과 함께 평균을 출력한다. 앞 `--n_vis`개(기본 10) 샘플의 예측 그림 `result_<순번>_sample<번호>.png`를 `--result_dir`에 저장한다.
- `--result_dir`의 기본값은 `./results`이다. 기준 결과 폴더 `results/`를 보존하기 위해 반드시 다른 경로를 지정한다.
- `--model`, `--size`는 체크포인트를 학습한 값과 같아야 한다. 조합은 `radiounet small|medium`, `transpathnet small|base`이다.
- 4750 MHz RadioUNet-S는 `checkpoints_radiounet_small_4750_v2/best.pth`(lr 3e-5로 학습)가 있으면 분석 스크립트가 이 체크포인트를 쓴다. 단일 평가에서 같은 값을 얻으려면 `--ckpt`에 v2 경로를 준다.

### 5.4 모델 학습

**실행 위치**: 호스트 · 작업 디렉터리: 패키지 루트

```bash
docker compose run --rm rmp python -m ikeee.bratlab_train --mode train \
    --root data --freq 2750 --model radiounet --size small \
    --batch_size 8 --epochs 100 --lr 1e-4 \
    --save_dir model_checkpoints/checkpoints_radiounet_small_2750 2>&1 | tee train_radiounet_small_2750.log
```

정상 출력 예(발췌)

```text
[DataLoader-Fixed] Train:1400 | Val:200 | Test:400
[Train] Device: cuda | Model: radiounet-small | Seed: 42
[Model] radiounet-small | Params: 34.92M
[Train] Batch size: 8 | Accum steps: 1 | AMP: False
[001/100] Train:0.xxxx Val:0.xxxx | RMSE:0.xxxx MAE:0.xxxx SSIM:0.xxxx | 95.3s
  ✅ Best saved → model_checkpoints/checkpoints_radiounet_small_2750/best.pth
...
[100/100] Train:0.xxxx Val:0.xxxx | RMSE:0.xxxx MAE:0.xxxx SSIM:0.xxxx | 94.8s
  완료 표시 저장 → model_checkpoints/checkpoints_radiounet_small_2750/training_complete.txt
[Train] radiounet-small 완료!
```

로그의 손실·지표 값(`x`로 표시)은 실행마다 다르다. 논문 결과(seed 42 체크포인트)에 사용한 학습 설정은 다음과 같다. 전체 옵션은 『코드 매뉴얼』 5.3에 있다.

| 모델 | `--model --size` | Batch size | Learning rate | Epoch | `best.pth` epoch(1750/4750/5750) |
|---|---|---|---|---|---|
| RadioUNet-S | `radiounet small` | 8 | 1e-4 | 100 | 97 / 84(v2: 99) / 98 |
| RadioUNet-M | `radiounet medium` | 4 | 1e-4 | 100 | 97 / 99 / 92 |
| TransPathNet-S | `transpathnet small` | 2 | 1e-4 | 100 | 96 / 97 / 95 |
| TransPathNet-B | `transpathnet base` | 2 | 1e-4 | 100 | 100 / 97 / 94 |

- 위 설정은 RTX 2080 Ti(11 GB)에서 학습한 값이다. `best.pth` epoch는 검증 손실이 최저였던 epoch로, 체크포인트의 `epoch` 키에 기록되어 있다.
- `--save_dir`에 `best.pth`(검증 손실이 가장 낮은 epoch의 가중치), `training_curve.png`(손실·지표 곡선), `training_complete.txt`(설정 요약)가 생긴다. `best.pth`는 학습 중 갱신되므로 중단해도 그때까지의 최선 가중치가 남는다.
- 체크포인트 폴더 이름을 `checkpoints_<모델>_<크기>_<주파수>`로 지으면 설치 확인과 분석 스크립트가 자동으로 찾는다.
- GPU 메모리가 부족하면 `--batch_size`를 줄이고 `--accum_steps`로 유효 batch를 유지한다(예: `--batch_size 2 --accum_steps 4`). `--amp`를 켜면 메모리와 시간이 줄지만 결과가 미세하게 달라진다.
- 같은 설정으로 다시 학습해도 GPU 연산 순서 때문에 가중치가 비트 단위로 같지는 않다. `--deterministic`을 켜면 차이가 줄지만 느려진다(『코드 매뉴얼』 8.3).
- 학습 중 다른 명령을 같은 패키지에서 실행해도 되지만 GPU 메모리를 함께 쓰므로 평가·재현은 학습이 끝난 뒤 수행한다.

### 5.5 반복 holdout 실험

분할 seed를 바꿔 가며 학습·평가·집계를 반복하는 스크립트이다. 실행할 명령만 보려면 `DRY_RUN=1`을 준다.

**실행 위치**: 호스트 · 작업 디렉터리: 패키지 루트

```bash
docker compose run --rm -e DRY_RUN=1 rmp bash scripts/run_multi_seed_holdout_experiments.sh
docker compose run --rm -e SEEDS="43 44" -e FREQS="4750" \
    -e TRANSPATHNET_BATCH_SIZE=2 -e TRANSPATHNET_ACCUM_STEPS=4 -e TRANSPATHNET_AMP=1 \
    rmp bash scripts/run_multi_seed_holdout_experiments.sh 2>&1 | tee multi_seed.log
```

정상 출력 예(`DRY_RUN=1`, 발췌)

```text
ROOT=/workspace/data
OUT_DIR=/workspace/results_multi_seed_holdout
SEEDS=42 43 44 45 46
FREQS=1750 4750 5750
MODELS=radiounet-small radiounet-medium transpathnet-small transpathnet-base
...
> python -m ikeee.bratlab_train --mode train --root /workspace/data --freq 1750 --model radiounet --size small --img_size 256 --batch_size 8 --accum_steps 1 --epochs 100 --lr 1e-4 --num_workers 4 --save_dir /workspace/results_multi_seed_holdout/seed_42/model_checkpoints/checkpoints_radiounet_small_1750 --seed 42
...
Done. Results root: /workspace/results_multi_seed_holdout
```

| 환경 변수 | 기본값 | 의미 |
|---|---|---|
| `SEEDS` | `42 43 44 45 46` | 분할·학습 seed 목록 |
| `FREQS` | `1750 4750 5750` | 주파수 목록 |
| `MODELS` | 4개 모델 태그 | 모델 목록(`radiounet-small` 등) |
| `EPOCHS`, `BATCH_SIZE`, `ACCUM_STEPS`, `AMP`, `LR` | `100`, `8`, `1`, `0`, `1e-4` | 학습 설정 |
| `TRANSPATHNET_BATCH_SIZE`, `TRANSPATHNET_ACCUM_STEPS`, `TRANSPATHNET_AMP` | `BATCH_SIZE` 등과 같음 | TransPathNet에만 적용하는 학습 설정 |
| `TRAIN`, `EVAL`, `AGGREGATE` | `1`, `1`, `1` | 단계 선택(0이면 생략) |
| `SKIP_EXISTING` | `1` | 완료된 체크포인트·지표가 있으면 건너뜀 |
| `OUT_DIR` | `results_multi_seed_holdout` | 결과 루트 |
| `DRY_RUN` | `0` | 1이면 명령만 출력 |

- 결과는 `results_multi_seed_holdout/seed_<seed>/model_checkpoints/`, `seed_<seed>/metrics/`, `aggregate/`(seed 평균·표준편차 표)에 저장된다.
- 논문의 seed 43, 44 실험은 TransPathNet을 batch 2, 누적 4, AMP로, RadioUNet을 batch 8, AMP 없이 학습했다. 기본값 그대로 실행하면 모든 모델이 batch 8로 학습되어 GPU 메모리 11 GB에서는 TransPathNet이 메모리 부족으로 실패한다.
- 전체 실행(seed 5개 × 주파수 3개 × 모델 4개 × 100 epoch)은 GPU 1대에서 수 주가 걸린다. 필요한 seed·주파수·모델만 지정한다.

### 5.6 결과 파일

**실행 위치**: 호스트 · 작업 디렉터리: 패키지 루트

```bash
ls results_1750_4750_5750_regenerated/metrics
head -3 results_1750_4750_5750_regenerated/metrics/summary_metrics_fixed_threshold.csv | cut -c1-120
```

정상 출력 예

```text
README_fixed_threshold_metrics.txt  heatmap_fixed_threshold_th-65.png  per_sample_metrics_fixed_threshold.csv  table_fixed_threshold_th-60.png  table_fixed_threshold_th-70.png
heatmap_fixed_threshold_th-60.png   heatmap_fixed_threshold_th-70.png  summary_metrics_fixed_threshold.csv     table_fixed_threshold_th-65.png
seed,threshold_dbm,freq_mhz,model_tag,model_label,n_samples,precision_mean,precision_std,recall_mean,recall_std,iou_mean,iou_std,f1_mean,f1_std,...
42,-70.0,1750,radiounet-small,RadioUNet-S (35M),400,0.8595971002346086,0.09292001940868291,0.7888198321480892,...
```

결과 CSV의 컬럼과 계산식, 그림의 구성은 『코드 매뉴얼』 6장과 부록 B에 있다.

## 6. 결과 관리

| 구분 | 위치 |
|---|---|
| 기준 결과(패키지 제공) | `results/representative_1750_4750_5750/{metrics, miss_cdf, boundary_cdf, qualitative}/`: 대표 결과 파이프라인의 논문 결과. `results/`의 다른 폴더는 논문 그림용 보조 결과(『코드 매뉴얼』 1.3) |
| 대표 결과 재현 | `results_1750_4750_5750_regenerated/`(또는 `OUT_DIR`). 실행 로그 `pipeline.log` |
| 단일 체크포인트 평가 | `--result_dir`로 지정한 폴더(본 문서는 `runs/<이름>/`). 지표는 화면에만 출력되므로 로그를 남긴다 |
| 학습 | `--save_dir`로 지정한 폴더의 `best.pth`, `training_curve.png`, `training_complete.txt` |
| 반복 holdout 실험 | `results_multi_seed_holdout/seed_<seed>/`, `aggregate/` |

- 재현과 학습은 같은 출력 폴더에 다시 실행하면 파일을 덮어쓴다. 보존할 결과는 다른 이름의 폴더로 옮기거나 `OUT_DIR`, `--result_dir`, `--save_dir`를 바꿔 실행한다.
- 결과를 보존할 때는 결과 폴더와 실행 로그, 사용한 체크포인트 폴더 이름(`training_complete.txt`에 설정이 있음)을 함께 보관한다.
- 결과 파일의 소유자는 `.env`의 UID/GID이다. 소유자가 root로 나오면 `.env`가 없거나 다른 사용자 값으로 생성된 것이므로 4.5.2를 다시 수행한다.

기준 결과와 비교하는 방법이다. 같은 체크포인트로 같은 분할을 평가하므로 값이 같아야 하며, GPU에 따른 차이는 TransPathNet 행의 소수점 6째 자리 이하에 한정된다.

**실행 위치**: 호스트 · 작업 디렉터리: 패키지 루트

```bash
R=results/representative_1750_4750_5750
N=results_1750_4750_5750_regenerated
diff <(cut -d, -f2- $N/metrics/summary_metrics_fixed_threshold.csv) $R/metrics/summary_metrics_fixed_threshold.csv | grep "^<" | cut -d, -f1-4
diff $N/boundary_cdf/boundary_cdf_summary.csv $R/boundary_cdf/boundary_cdf_summary.csv | grep "^<" | cut -d, -f1-3
diff $N/qualitative/selected_shared_sample.csv $R/qualitative/selected_shared_sample.csv | grep "^<" | cut -d, -f1-2
```

정상 출력 예(지표 요약, 발췌)

```text
< -70.0,1750,transpathnet-small,TransPathNet-S (38M)
< -70.0,1750,transpathnet-base,TransPathNet-B (111M)
< -70.0,4750,transpathnet-small,TransPathNet-S (38M)
...
```

- 다른 GPU에서 재현하면 지표 요약 36행 중 TransPathNet 행 18개만 다르게 나오고, 그 차이는 소수점 6째 자리 이하이다. RadioUNet 행이 다르거나 차이가 그보다 크면 체크포인트나 데이터셋이 다른 것이다.
- 경계 영역 요약과 정성 그림의 선택 샘플도 같은 방식으로 비교한다. TransPathNet 값이 들어간 행만 미세하게 다르고, 선택 샘플은 770으로 같아야 한다.
- 같은 GPU에서 두 번 실행하면 모든 행이 같다.

## 7. 문제 해결

### 7.1 GPU 메모리 부족

`torch.cuda.OutOfMemoryError: CUDA out of memory`가 나오면 다음 순서로 조치한다.

- 학습: `--batch_size`를 절반으로 줄이고 `--accum_steps`를 두 배로 늘린다. 그래도 부족하면 `--amp`를 켠다. GTX 1050(4 GB)에서는 RadioUNet-S를 `--batch_size 4 --amp`로 학습할 수 있다.
- 평가·대표 결과 재현: batch size가 1로 고정되어 있어 4 GB GPU에서도 네 모델이 모두 동작한다. 그래도 부족하면 다른 프로세스의 GPU 사용을 끝내거나 `rmp-cpu`로 실행한다.
- 컨테이너는 끝나면 GPU 메모리를 반환한다. `nvidia-smi`로 다른 프로세스가 메모리를 쓰고 있는지 확인한다.

### 7.2 오류 메시지와 조치

| 메시지 | 원인 | 조치 |
|---|---|---|
| `permission denied while trying to connect to the Docker daemon socket` | Docker 권한 없음 | 4.3.3 |
| `could not select device driver "nvidia" with capabilities: [[gpu]]` | NVIDIA Container Toolkit 미설치 또는 Docker 재시작 누락 | 4.3.4. Windows는 Docker Desktop의 WSL integration 확인(B.1) |
| `[ERROR] Dataset folder not found: ...` | `setup_env.sh`에 준 데이터셋 경로 오류 | 경로 확인 후 4.5.2 다시 수행 |
| `Result: FAIL (see MISSING / FAIL lines above)` | 패키지·코드·데이터셋 중 하나가 없음 | MISSING 줄의 항목 확인. 데이터셋은 4.4, 이미지는 4.5.1 |
| `FileNotFoundError: Missing checkpoint: /workspace/model_checkpoints/...` | 체크포인트가 없거나 폴더 이름이 다름 | 4.4로 체크포인트 배치. 직접 학습한 경우 폴더 이름을 `checkpoints_<모델>_<크기>_<주파수>`로 맞춤 |
| `[SKIP] Missing checkpoint`(경계 영역 단계) | 같은 원인. 이 단계는 없는 모델을 건너뜀 | 위와 같음. 건너뛴 모델은 결과에 빠짐 |
| `RuntimeError: Sample id N not found in fixed test loader` | 정성 그림의 `--sample_id`가 테스트 분할에 없음 | `SAMPLE_ID=auto`(기본) 사용 또는 테스트 분할의 샘플 번호 지정 |
| `DataLoader worker (pid N) is killed by signal: Bus error` | 컨테이너 공유 메모리 부족 | `docker-compose.yml`의 `shm_size`(기본 8gb) 증가 또는 `--num_workers` 감소 |
| `CUDA out of memory` | GPU 메모리 부족 | 7.1 |
| `UserWarning: Applied workaround for CuDNN issue, install nvrtc.so` | PyTorch 2.2 CUDA 라이브러리 경고 | 무시. 결과에 영향 없음 |
| `groups: cannot find name for group ID 1000`, 프롬프트 `I have no name!` | 컨테이너 안에 호스트 사용자 이름이 없음 | 무시. UID/GID만 맞으면 됨 |
| 결과 파일 소유자가 root | `.env` 없음 또는 UID/GID 불일치 | 4.5.2 다시 수행. 기존 파일은 `sudo chown -R $USER:$USER <폴더>` |
| `error: unable to prepare context` 또는 빌드 중 다운로드 실패 | 경로 B에서 인터넷 연결 없음 | 경로 A(제공 이미지) 사용 |
| `[Warning] pytorch_msssim 없음` | 부록 C 환경에서 패키지 누락 | `pip install pytorch-msssim==1.0.0`. 없으면 SSIM 항이 빠져 손실·지표가 달라짐 |

## 8. 코드 변경과 재실행

`ikeee/`, `scripts/`의 코드를 고친 뒤 다음 순서로 반영을 확인한다. 호스트의 패키지 루트가 컨테이너의 `/workspace`에 연결되므로 이미지를 다시 만들 필요가 없다.

**실행 위치**: 호스트 · 작업 디렉터리: 패키지 루트

```bash
docker compose run --rm rmp python scripts/check_setup.py
docker compose run --rm rmp python -c "import ikeee.bratlab_train, ikeee.bratlab_dataset; print('import OK')"
docker compose run --rm rmp python -m ikeee.bratlab_train --mode eval --root data --freq 4750 --model radiounet --size small \
    --ckpt model_checkpoints/checkpoints_radiounet_small_4750/best.pth --result_dir runs/after_change --n_vis 3
docker compose run --rm -e OUT_DIR=/workspace/runs/pipeline_after_change rmp bash scripts/run_1750_4750_5750_shared_pipeline.sh
```

- `docker/requirements.txt`나 `Dockerfile`을 바꾼 경우에만 `docker compose build`(4.6)로 이미지를 다시 만든다.
- 모델 구조를 바꾸면 기존 체크포인트와 맞지 않아 `load_state_dict` 오류가 난다. 바꾼 구조로 다시 학습한다(5.4).
- 데이터셋 로더나 분할 규칙을 바꾸면 테스트 분할이 달라져 기준 결과와 비교할 수 없다. 변경 전 결과를 같은 환경에서 다시 만들어 비교 기준으로 쓴다.
- 코드 수정 위치와 새 모델 추가 순서는 『코드 매뉴얼』 7장에 있다.

## 9. 디스크 정리

실행 결과와 설정 파일을 지우고 처음 받은 상태로 되돌리는 방법이다.

**실행 위치**: 호스트 · 작업 디렉터리: 패키지 루트

```bash
rm -rf results_1750_4750_5750_regenerated results_multi_seed_holdout runs .env *.log
rm -rf model_checkpoints/checkpoints_test_*
find . -name __pycache__ -prune -exec rm -rf {} +
sha256sum --quiet -c SHA256SUMS && echo "package files unchanged"
```

- 보존할 결과는 지우기 전에 다른 위치로 옮긴다.
- 직접 학습한 체크포인트는 `model_checkpoints/` 아래에 있으므로 필요한 것만 골라 지운다. 제공 체크포인트는 `LIG_RMP_SHADOW_v1.0_checkpoints.tar`에서 다시 풀 수 있다.
- 마지막 명령이 `package files unchanged`이면 패키지 파일은 받은 상태와 같다.
- 환경 이미지까지 지우려면 `docker image rm outdoor-rmp:latest`을 실행한다. 다시 쓰려면 4.5.1부터 수행한다. 데이터셋·체크포인트를 지우려면 `rm -rf data/* model_checkpoints/*`를 실행하고 4.4에서 다시 푼다.

## 부록 A. 컨테이너 셸 직접 사용

명령마다 컨테이너를 만들지 않고 셸로 들어가 여러 명령을 실행하는 방법이다.

**실행 위치**: 호스트 → 컨테이너 · 작업 디렉터리: 패키지 루트(컨테이너 안에서는 `/workspace`)

```bash
docker compose run --rm rmp bash
python scripts/check_setup.py
python -m ikeee.bratlab_train --help
python scripts/final_eval_metrics_fixed_threshold_heatmap.py --help
nvidia-smi
exit
```

정상 출력 예(일부)

```text
I have no name!@6d3eb6ef4029:/workspace$ python -m ikeee.bratlab_train --help
usage: bratlab_train.py [-h] [--mode {train,eval}] --root ROOT [--freq {1750,2750,3750,4750,5750}] [--model {transpathnet,radiounet}] [--size {small,medium,base}] ...
```

- 프롬프트의 `I have no name!`은 컨테이너에 호스트 사용자 이름이 등록되지 않아 나오는 표시이며 동작에는 영향이 없다.
- 컨테이너 안에서 만든 파일은 `/workspace` 아래이면 호스트의 패키지 안에 남고, 그 밖의 위치(`/tmp` 등)는 `exit`하면 사라진다.
- 셸을 나오면(`exit`) 컨테이너는 삭제된다.

## 부록 B. Windows(WSL 2 + Docker Desktop)에서 설치

Windows 11 PC에서는 Docker Engine 대신 Docker Desktop을 설치하고, WSL 2의 Ubuntu 터미널에서 4.4 이후의 명령을 그대로 실행한다. 본 패키지의 검증은 이 환경에서 수행했다(B.4).

### B.1 사전 환경

1. WSL 2와 Ubuntu 설치: PowerShell(관리자 권한)에서 `wsl --install -d Ubuntu`를 실행하고 재부팅한다. Ubuntu 창이 열리면 사용자 이름과 비밀번호를 정한다. `wsl -l -v`에서 VERSION이 2이어야 한다.
2. NVIDIA 드라이버: Windows용 NVIDIA 드라이버만 설치한다. WSL 안에는 드라이버를 설치하지 않는다. PowerShell에서 `nvidia-smi`가 동작하면 된다.
3. Docker Desktop: `https://www.docker.com/products/docker-desktop`에서 설치한다(WSL 2 backend). 실행 후 Settings → Resources → WSL integration에서 Ubuntu를 켜고 Apply & restart 한다. NVIDIA Container Toolkit은 Docker Desktop에 포함되어 있어 설치하지 않으며, 4.3.3의 docker 그룹 추가도 필요 없다.

**실행 위치**: WSL Ubuntu 터미널 · 작업 디렉터리: 임의

```bash
docker version
docker compose version
docker info --format '{{.OSType}}/{{.Architecture}} {{.ServerVersion}}'
```

정상 출력 예

```text
Client: Docker Engine - Community
 Version:           29.4.3
...
Server: Docker Desktop
 Engine:
  Version:          29.4.3
...
Docker Compose version v5.1.3
linux/x86_64 29.4.3
```

- `The command 'docker' could not be found in this WSL 2 distro`가 나오면 WSL integration이 꺼진 것이다. Docker Desktop 설정에서 켠 뒤 Ubuntu 터미널을 다시 연다.
- PowerShell에서 `wsl`만 치면 `docker-desktop` 배포판이 열릴 수 있다(프롬프트 `docker-desktop:~#`). 이 배포판은 Docker Desktop 내부용이므로 `wsl -d Ubuntu`로 Ubuntu를 열거나 `wsl --set-default Ubuntu`로 기본 배포판을 바꾼다.

### B.2 파일 복사와 작업 위치

- 패키지, 데이터셋, 체크포인트는 WSL 홈 디렉터리(예: `/home/<user>/rmp-work`) 아래에 둔다. `/mnt/c` 아래의 Windows 경로는 파일 접근이 느리고 권한 문제가 생긴다.

**실행 위치**: WSL Ubuntu 터미널 · 작업 디렉터리: 홈 디렉터리

```bash
cp /mnt/c/Users/<Windows 사용자>/Downloads/LIG_RMP_SHADOW_v1.0*.tar* ~/
ls -l ~/*.tar ~/*.sha256
explorer.exe .          # 현재 WSL 폴더를 Windows 탐색기로 열기(끌어다 놓기로 복사할 때)
```

- Windows 탐색기 주소창에 `\\wsl$\Ubuntu\home\<user>`를 입력해도 WSL 홈이 열린다. 탐색기로 복사하면 `<파일>:Zone.Identifier`라는 파일이 같이 생길 수 있으며 지워도 된다.
- WSL의 가상 디스크는 Windows C 드라이브의 공간을 쓴다. 패키지·이미지·데이터셋·체크포인트에 50 GB 이상이 필요하다(3.2).

### B.3 설치와 실행

4.4(압축 해제와 무결성 확인) → 4.5(경로 A) → 5장(실행)을 WSL Ubuntu 터미널에서 그대로 수행한다. 다음 두 가지가 Linux 호스트와 다르다.

- 4.3(Docker 준비)은 수행하지 않는다. B.1의 확인으로 대신한다.
- GPU 메모리가 4 GB 이하인 PC에서는 학습 시 `--batch_size 4 --amp`(RadioUNet-S) 이하로 설정한다. 평가와 대표 결과 재현은 batch size 1이므로 그대로 동작한다.
- Docker Desktop은 `docker images`에서 이미지 크기를 DISK USAGE(약 12.5 GB)와 CONTENT SIZE(약 4.1 GB)로 나누어 보여 준다.
- WSL 터미널을 닫으면 실행 중인 명령이 중단된다. 긴 작업은 터미널을 닫지 않거나 Windows 터미널의 새 탭에서 다른 작업을 한다.

### B.4 검증 환경과 결과

| 항목 | 내용 |
|---|---|
| Windows | Windows 11 25H2(OS 빌드 26200) |
| WSL | WSL 2, Ubuntu 26.04 LTS |
| Docker | Docker Desktop, Docker Engine 29.4.3, Docker Compose v5.1.3 |
| GPU | NVIDIA GeForce GTX 1050(4 GB), 드라이버 560.94 |
| 환경 이미지 | `outdoor-rmp:latest`, 이미지 ID `sha256:377190948642…`(경로 B로 빌드한 뒤 `docker save`로 저장한 것이 제공 이미지 파일) |

| 수행 항목 | 결과 |
|---|---|
| 설치 확인(`check_setup.py`) | `Result: OK`. 패키지 7개, GPU, 코드, 데이터셋 5개 주파수 × 2000 파일, 체크포인트 13개 인식 |
| 테스트 데이터 1 epoch 학습·평가(4.5.5) | RadioUNet-S, batch 4, AMP. 학습·평가·그림 저장 정상 |
| 대표 결과 재현(5.2) | 네 단계 모두 완료. `summary_metrics_fixed_threshold.csv`의 RadioUNet 행은 기준 결과와 동일, TransPathNet 행은 소수점 6째 자리 이하 차이. 공통 샘플 770 선택 |

## 부록 C. Docker 없이 직접 설치

Docker를 쓸 수 없는 Linux 호스트에서 Python 가상 환경에 직접 설치하는 방법이다. 논문 결과를 만든 환경(Ubuntu 20.04, Python 3.11, PyTorch 2.2.0 + CUDA 11.8)과 같은 구성이다.

### C.1 패키지 설치

**실행 위치**: 호스트 · 작업 디렉터리: 패키지 루트

```bash
python3 --version                      # 3.10 또는 3.11
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2" scipy==1.15.3 matplotlib==3.10.0 pillow pytorch-msssim==1.0.0
python scripts/check_setup.py --root data --ckpt_dir model_checkpoints
```

- NVIDIA 드라이버는 CUDA 11.8을 지원해야 한다(520 이상). `check_setup.py`의 `[2/5] GPU`에 GPU 이름이 나오면 된다.
- `requirements.txt`는 버전을 고정하지 않은 목록이다. 결과를 재현하려면 위와 같이 버전을 지정해 설치한다.

### C.2 실행

5장의 명령에서 `docker compose run --rm rmp`를 빼고 패키지 루트에서 직접 실행한다. 데이터셋과 체크포인트 경로는 `.env` 대신 명령 인자로 준다.

**실행 위치**: 호스트 · 작업 디렉터리: 패키지 루트(가상 환경 활성화 상태)

```bash
export PYTHONPATH=$(pwd)
ROOT=$(pwd)/data CKPT_DIR=$(pwd)/model_checkpoints bash scripts/run_1750_4750_5750_shared_pipeline.sh 2>&1 | tee pipeline.log
python -m ikeee.bratlab_train --mode eval --root data --freq 4750 --model radiounet --size small \
    --ckpt model_checkpoints/checkpoints_radiounet_small_4750/best.pth --result_dir runs/eval_check
```

- 패키지 버전이 다르면 TransPathNet 결과가 소수점 아래에서 달라질 수 있다. 비교 기준은 같은 환경에서 다시 만든다(6장).
- matplotlib가 화면 없는 서버에서 오류를 내면 `export MPLBACKEND=Agg`를 먼저 실행한다.
