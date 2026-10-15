"""
emcad_decoder.py

EMCAD: Efficient Multi-scale Convolutional Attention Decoder
원본: Rahman et al., CVPR 2024 (BSD 3-Clause)
참고: https://github.com/SLDGroup/EMCAD

TransPathNet에서 사용하는 decoder head.
다중 스케일에서 convolutional attention으로 feature를 정제하고
최종 pathloss map을 생성.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────
# MSCAM: Multi-Scale Convolutional Attention Module
# ─────────────────────────────────────────────────────

class MSCAM(nn.Module):
    """
    Multi-Scale Convolutional Attention Module.
    다양한 커널 크기의 depthwise conv로
    다중 스케일 feature를 병렬 추출 후 attention 적용.
    """
    def __init__(self, ch, kernels=(1, 3, 5, 7)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, k, padding=k // 2,
                          groups=ch, bias=False),
                nn.BatchNorm2d(ch),
                nn.GELU(),
            )
            for k in kernels
        ])
        # Channel attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch * len(kernels), ch),
            nn.ReLU(),
            nn.Linear(ch, ch * len(kernels)),
            nn.Sigmoid(),
        )
        self.fuse = nn.Conv2d(ch * len(kernels), ch, 1)

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        cat   = torch.cat(feats, dim=1)             # (B, C*K, H, W)
        # Channel attention weight
        w = self.ca(cat).reshape(
            cat.shape[0], cat.shape[1], 1, 1)
        out = self.fuse(cat * w)
        return out + x                               # residual


# ─────────────────────────────────────────────────────
# LGAG: Local-Global Attention Gate
# ─────────────────────────────────────────────────────

class LGAG(nn.Module):
    """
    Local-Global Attention Gate.
    encoder skip feature와 decoder feature를 결합할 때
    중요한 공간 위치를 선택적으로 강조.
    """
    def __init__(self, ch_g, ch_x):
        """
        ch_g: decoder (gating) feature 채널
        ch_x: encoder skip feature 채널
        """
        super().__init__()
        inter = max(ch_g // 4, 16)
        self.W_g  = nn.Conv2d(ch_g, inter, 1)
        self.W_x  = nn.Conv2d(ch_x, inter, 1)
        self.psi  = nn.Sequential(
            nn.Conv2d(inter, 1, 1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g: decoder feature (gating signal)
        x: encoder skip feature
        """
        # 크기 맞추기
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:],
                              mode='bilinear', align_corners=False)
        att = self.relu(self.W_g(g) + self.W_x(x))
        att = self.psi(att)
        return x * att


# ─────────────────────────────────────────────────────
# EMCAD Decoder Block (업샘플 1단계)
# ─────────────────────────────────────────────────────

class EMCADBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        # Upsampling
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        # Attention gate for skip connection
        self.gate = LGAG(out_ch, skip_ch)
        # Multi-scale conv attention
        self.mscam = MSCAM(out_ch + skip_ch)
        # Fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x, skip):
        x    = self.up(x)
        # 크기 불일치 보정
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode='bilinear', align_corners=False)
        skip = self.gate(x, skip)                    # attention gate
        out  = self.mscam(torch.cat([x, skip], 1))  # multi-scale attention
        return self.fuse(out)


# ─────────────────────────────────────────────────────
# EMCAD Decoder (전체)
# ─────────────────────────────────────────────────────

class EMCADDecoder(nn.Module):
    """
    4단계 TransNeXt feature pyramid → 1채널 pathloss map

    입력 feature pyramid (TransNeXt 출력):
        C1: (B, 64,  H/4,  W/4)
        C2: (B, 128, H/8,  W/8)
        C3: (B, 256, H/16, W/16)
        C4: (B, 512, H/32, W/32)

    출력:
        (B, 1, H, W)  ← 원본 해상도
    """

    def __init__(self, embeds=(64, 128, 256, 512)):
        super().__init__()
        e1, e2, e3, e4 = embeds

        # C4 → C3 레벨
        self.d3 = EMCADBlock(e4, e3, e3)
        # C3 → C2 레벨
        self.d2 = EMCADBlock(e3, e2, e2)
        # C2 → C1 레벨
        self.d1 = EMCADBlock(e2, e1, e1)

        # C1 → 원본 해상도 (×4 업샘플)
        self.up_final = nn.Sequential(
            nn.ConvTranspose2d(e1, e1 // 2, 2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(e1 // 2, e1 // 4, 2, stride=2),
            nn.GELU(),
        )

        # 최종 예측 헤드
        self.head = nn.Sequential(
            nn.Conv2d(e1 // 4, e1 // 4, 3, padding=1),
            nn.BatchNorm2d(e1 // 4),
            nn.GELU(),
            nn.Conv2d(e1 // 4, 1, 1),
            nn.Sigmoid(),   # 출력: [0, 1]
        )

        # 보조 출력 헤드 (중간 스케일, 학습 안정화)
        self.aux_head = nn.Sequential(
            nn.Conv2d(e2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, c1, c2, c3, c4):
        x = self.d3(c4, c3)         # (B, 256, H/16, W/16)
        x = self.d2(x,  c2)         # (B, 128, H/8,  W/8)

        # 보조 출력 (학습 시 deep supervision용)
        aux = F.interpolate(self.aux_head(x),
                            scale_factor=8, mode='bilinear',
                            align_corners=False)

        x   = self.d1(x, c1)        # (B, 64,  H/4,  W/4)
        x   = self.up_final(x)      # (B, 16,  H,    W)
        out = self.head(x)          # (B, 1,   H,    W)

        return out, aux


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = EMCADDecoder().to(device)
    c1 = torch.randn(2,  64, 64, 64).to(device)
    c2 = torch.randn(2, 128, 32, 32).to(device)
    c3 = torch.randn(2, 256, 16, 16).to(device)
    c4 = torch.randn(2, 512,  8,  8).to(device)
    out, aux = decoder(c1, c2, c3, c4)
    print(f"Output : {out.shape}")   # (2, 1, 256, 256)
    print(f"Aux    : {aux.shape}")   # (2, 1, 256, 256)