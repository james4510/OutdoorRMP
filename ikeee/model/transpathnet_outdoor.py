"""
transpathnet_outdoor.py

TransPathNet Outdoor Edition
- 원본: LiXin97/TransPathNet (실내 전용, ICASSP 2025 Winner)
- 수정: 실외 BRAT-Lab 데이터셋에 맞게 입력 채널 재설계

=================================================================
[원본 실내 입력 채널 구성]                [실외 변경 사항]
─────────────────────────────────────────────────────────────────
Ch0: Reflectance map          →  Ch0: Building mask (outdoor)
Ch1: Transmittance map        →  Ch1: Tx position map (outdoor)
Ch2: Distance map             →  제거
Ch3: FSPL map                 →  제거
Ch4: Transmission ray         →  제거 (실내 고유 특징)
Ch5: Antenna embedding        →  제거 (단일 안테나 가정)
Ch6: Spatial-freq embedding   →  제거
=================================================================

실외 입력 채널 (2ch):
    Ch0: Building mask    - 건물 위치 (0/1)
    Ch1: Tx position map  - 송신기 위치 맵

2단계 구조:
    Coarse Stage: 2ch → Coarse Radio Map (1ch)
    Fine Stage:   2ch + 1ch(coarse) = 3ch → Fine Radio Map (1ch)

모델 크기:
    small : embeds=(64,128,256,512),  depths=(2,2,6,2)  → ~38.10M
    base  : embeds=(96,192,384,768),  depths=(3,3,9,3)  → ~111.45M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ikeee.model.transnext_backbone import TransNeXtBackbone
from ikeee.model.emcad_decoder import EMCADDecoder


# ─────────────────────────────────────────────────────
# Feature Extractor Unit (Backbone + Decoder 묶음)
# ─────────────────────────────────────────────────────

class TransPathUnit(nn.Module):
    """
    TransNeXt Backbone + EMCAD Decoder 한 쌍.
    Coarse / Fine stage 각각에 사용.
    """
    def __init__(self, in_ch: int,
                 embeds=(64, 128, 256, 512),
                 depths=(2, 2, 6, 2),
                 heads=(2, 4, 8, 16)):
        super().__init__()
        self.backbone = TransNeXtBackbone(
            in_ch=in_ch, embeds=embeds,
            depths=depths, heads=heads,
        )
        self.decoder = EMCADDecoder(embeds=embeds)

    def forward(self, x):
        c1, c2, c3, c4 = self.backbone(x)
        out, aux = self.decoder(c1, c2, c3, c4)
        return out, aux


# ─────────────────────────────────────────────────────
# TransPathNet Outdoor (전체 2단계 모델)
# ─────────────────────────────────────────────────────

class TransPathNetOutdoor(nn.Module):
    """
    TransPathNet Outdoor Edition for BRAT-Lab Dataset.

    입력:
        x: (B, 2, H, W)
            Ch0: Building mask
            Ch1: Tx position map

    출력:
        coarse: (B, 1, H, W) - 1단계 예측
        fine:   (B, 1, H, W) - 2단계 최종 예측
        aux:    (B, 1, H, W) - 보조 출력 (학습 안정화)

    2단계 구조:
        [Coarse Stage]
            입력: 2ch
            출력: coarse map (1ch)
        [Fine Stage]
            입력: 2ch + coarse(1ch) = 3ch
            출력: fine map (1ch)
    """

    def __init__(
        self,
        embeds: tuple = (64, 128, 256, 512),
        depths: tuple = (2, 2, 6, 2),
        heads:  tuple = (2, 4, 8, 16),
    ):
        super().__init__()

        # Coarse stage: 2ch 입력
        self.coarse_net = TransPathUnit(
            in_ch=2, embeds=embeds, depths=depths, heads=heads
        )

        # Fine stage: 2ch + 1ch(coarse) = 3ch 입력
        self.fine_net = TransPathUnit(
            in_ch=3, embeds=embeds, depths=depths, heads=heads
        )

    def forward(self, x):
        """
        x: (B, 2, H, W)
        """
        # ── Coarse Stage ──────────────────────────
        coarse, aux_c = self.coarse_net(x)                        # (B, 1, H, W)

        # ── Fine Stage ────────────────────────────
        x_fine = torch.cat([x, coarse.detach()], dim=1)           # (B, 3, H, W)
        fine, aux_f = self.fine_net(x_fine)                       # (B, 1, H, W)

        return coarse, fine, aux_f


# ─────────────────────────────────────────────────────
# 모델 크기 설정 (Small / Base)
# ─────────────────────────────────────────────────────

MODEL_CONFIGS = {
    # 기본 설정 → ~38.10M  (RadioUNet-Small ~34.92M 과 비교)
    "small": dict(
        embeds=(64, 128, 256, 512),
        depths=(2, 2, 6, 2),
        heads=(2, 4, 8, 16),
    ),
    # 대용량 설정 → ~111.45M  (RadioUNet-Medium ~96.98M 과 비교)
    "base": dict(
        embeds=(96, 192, 384, 768),
        depths=(3, 3, 9, 3),
        heads=(3, 6, 12, 24),
    ),
}


def build_transpathnet(size: str = "small"):
    """
    size: 'small' | 'base'
    """
    assert size in MODEL_CONFIGS, f"size must be one of {list(MODEL_CONFIGS.keys())}"
    cfg = MODEL_CONFIGS[size]
    return TransPathNetOutdoor(**cfg)


# ─────────────────────────────────────────────────────
# 모델 정보 출력
# ─────────────────────────────────────────────────────

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    for size in ["small", "base"]:
        model = build_transpathnet(size).to(device)
        x     = torch.randn(2, 2, 256, 256).to(device)

        coarse, fine, aux = model(x)
        total, train = count_params(model)

        print(f"[TransPathNet-{size.upper()}]")
        print(f"  Input  : {x.shape}")
        print(f"  Coarse : {coarse.shape}")
        print(f"  Fine   : {fine.shape}")
        print(f"  Aux    : {aux.shape}")
        print(f"  Params : {train/1e6:.2f}M")
        print()