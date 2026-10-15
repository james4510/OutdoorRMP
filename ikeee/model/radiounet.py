"""
radiounet.py

RadioUNet (2-Stage, 2ch 입력)
원본: Levie et al., IEEE TWC 2021, arXiv:1911.09002

변경점:
    - 입력 채널: 2ch (Building mask + Tx position map)
    - base_ch, depth를 인자로 받아 Small / Medium 구성 지원
        Small  : base_ch=48,  depth=4  → ~34.92M
        Medium : base_ch=80,  depth=4  → ~96.98M
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────
# 기본 블록
# ─────────────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────────────
# 단일 UNet 모듈
# ─────────────────────────────────────────────────────

class UNetModule(nn.Module):
    """
    base_ch, depth를 인자로 받는 범용 UNet 모듈.

    채널 구성 (depth=4, base_ch=48 예시):
        Encoder   : 48 → 96 → 192 → 384
        Bottleneck: 384 → 768
        Decoder   : 768 → 384 → 192 → 96 → 48
        Head      : 48 → 1
    """

    def __init__(self, in_ch: int, base_ch: int, depth: int):
        super().__init__()

        # 각 단계별 채널 수: [base_ch, base_ch*2, ..., base_ch*2^(depth-1)]
        chs = [base_ch * (2 ** i) for i in range(depth)]

        # ── Encoder ──────────────────────────────────
        self.encoders = nn.ModuleList()
        self.encoders.append(DoubleConv(in_ch, chs[0]))
        for i in range(1, depth):
            self.encoders.append(DoubleConv(chs[i - 1], chs[i]))
        self.pool = nn.MaxPool2d(2)

        # ── Bottleneck ────────────────────────────────
        self.bottleneck = DoubleConv(chs[-1], chs[-1] * 2)

        # ── Decoder ──────────────────────────────────
        self.ups      = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            up_in = chs[-1] * 2 if i == depth - 1 else chs[i + 1]
            self.ups.append(nn.ConvTranspose2d(up_in, chs[i], 2, stride=2))
            self.decoders.append(DoubleConv(chs[i] * 2, chs[i]))

        # ── 출력 헤드 ─────────────────────────────────
        self.head = nn.Sequential(
            nn.Conv2d(chs[0], 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encoder + skip 저장
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder + skip connection
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = dec(torch.cat([up(x), skip], dim=1))

        return self.head(x)


# ─────────────────────────────────────────────────────
# RadioUNet (2-Stage: Coarse → Fine)
# ─────────────────────────────────────────────────────

class RadioUNet(nn.Module):
    """
    RadioUNet 2단계 구조.

    입력:
        x: (B, 2, H, W)
            Ch0: Building mask
            Ch1: Tx position map

    출력 (TransPathNet과 형식 통일):
        coarse: (B, 1, H, W) — 1단계 예측
        fine:   (B, 1, H, W) — 2단계 최종 예측
        aux:    fine와 동일  — loss 계산 형식 통일용
    """

    def __init__(self, in_ch: int = 2, base_ch: int = 48, depth: int = 4):
        super().__init__()

        # Stage 1: Coarse UNet (2ch 입력)
        self.coarse_unet = UNetModule(in_ch=in_ch,     base_ch=base_ch, depth=depth)

        # Stage 2: Fine UNet (2ch + 1ch coarse = 3ch 입력)
        self.fine_unet   = UNetModule(in_ch=in_ch + 1, base_ch=base_ch, depth=depth)

    def forward(self, x):
        # Stage 1
        coarse = self.coarse_unet(x)                              # (B, 1, H, W)

        # Stage 2: 원본 입력 + Coarse 결과 concat
        x_fine = torch.cat([x, coarse.detach()], dim=1)           # (B, 3, H, W)
        fine   = self.fine_unet(x_fine)                           # (B, 1, H, W)

        return coarse, fine, fine   # aux = fine (형식 통일)


# ─────────────────────────────────────────────────────
# 모델 크기 설정
# ─────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "small":  dict(base_ch=48, depth=4),   # ~34.92M
    "medium": dict(base_ch=80, depth=4),   # ~96.98M
}


def build_radiounet(size: str = "small"):
    """
    size: 'small' | 'medium'
    """
    assert size in MODEL_CONFIGS, f"size must be one of {list(MODEL_CONFIGS.keys())}"
    cfg = MODEL_CONFIGS[size]
    return RadioUNet(in_ch=2, **cfg)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train


# ─────────────────────────────────────────────────────
# 확인용
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    for size in ["small", "medium"]:
        model = build_radiounet(size).to(device)
        x     = torch.randn(2, 2, 256, 256).to(device)

        coarse, fine, aux = model(x)
        total, train = count_params(model)

        print(f"[RadioUNet-{size.upper()}]")
        print(f"  base_ch : {MODEL_CONFIGS[size]['base_ch']}")
        print(f"  depth   : {MODEL_CONFIGS[size]['depth']}")
        print(f"  Input   : {x.shape}")
        print(f"  Coarse  : {coarse.shape}")
        print(f"  Fine    : {fine.shape}")
        print(f"  Params  : {train/1e6:.2f}M")
        print()