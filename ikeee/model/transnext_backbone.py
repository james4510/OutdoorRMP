"""
transnext_backbone.py

TransNeXt Backbone (경량화 버전)
원본: https://github.com/DaiShiResearch/TransNeXt (Apache 2.0)

원본 TransNeXt는 ImageNet pretrained weight 기반의 대형 모델이나,
BRAT-Lab outdoor 환경에서는 from-scratch 학습을 위해
동일한 구조적 특징(Convolutional Attention + Swattention)을 유지하되
파라미터를 줄인 버전으로 구현.

핵심 특징:
- Aggregated Attention (local conv + global attention 결합)
- Pixel-focused attention (foveal visual perception)
- 4단계 계층적 feature 추출 (FPN 호환)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────
# 기본 블록
# ─────────────────────────────────────────────────────

class DWConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, kernel, stride, padding,
                            groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class MLP(nn.Module):
    """Feed-Forward Network"""
    def __init__(self, ch, expansion=4, drop=0.0):
        super().__init__()
        hidden = int(ch * expansion)
        self.net = nn.Sequential(
            nn.Linear(ch, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, ch),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────
# Aggregated Attention (TransNeXt 핵심)
# local depthwise conv + global attention 결합
# ─────────────────────────────────────────────────────

class AggregatedAttention(nn.Module):
    """
    TransNeXt의 핵심 Aggregated Attention.
    - Local: Depthwise conv로 근거리 feature 수집
    - Global: Linear attention으로 장거리 의존성 모델링
    - 두 결과를 채널 단위 합산
    """
    def __init__(self, ch, num_heads=8, window_size=3, drop=0.0):
        super().__init__()
        assert ch % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = ch // num_heads
        self.scale     = self.head_dim ** -0.5
        self.window    = window_size

        # QKV projection
        self.qkv  = nn.Linear(ch, ch * 3, bias=False)
        self.proj = nn.Linear(ch, ch)
        self.drop = nn.Dropout(drop)

        # Local conv branch (foveal)
        self.local_conv = nn.Conv2d(ch, ch, window_size,
                                    padding=window_size // 2,
                                    groups=ch, bias=False)
        self.local_norm = nn.BatchNorm2d(ch)

        # Mixing weight
        self.mix = nn.Parameter(torch.ones(1))

    def forward(self, x, H, W):
        B, N, C = x.shape   # N = H*W

        # ── Global attention branch ──────────────────
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                    self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Efficient linear attention (O(N) approximation)
        k = F.softmax(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)
        global_out = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # ── Local conv branch (foveal perception) ────
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)
        local_out = self.local_norm(self.local_conv(x_2d))
        local_out = local_out.reshape(B, C, N).transpose(1, 2)

        # ── 결합 ─────────────────────────────────────
        out = global_out + torch.sigmoid(self.mix) * local_out
        return self.proj(out)


# ─────────────────────────────────────────────────────
# TransNeXt Block
# ─────────────────────────────────────────────────────

class TransNeXtBlock(nn.Module):
    def __init__(self, ch, num_heads=8, mlp_ratio=4,
                 drop=0.0, drop_path=0.0):
        super().__init__()
        self.norm1  = nn.LayerNorm(ch)
        self.norm2  = nn.LayerNorm(ch)
        self.attn   = AggregatedAttention(ch, num_heads, drop=drop)
        self.mlp    = MLP(ch, mlp_ratio, drop)
        # Stochastic depth
        self.dp     = nn.Identity() if drop_path == 0 else \
                      StochasticDepth(drop_path)

    def forward(self, x, H, W):
        x = x + self.dp(self.attn(self.norm1(x), H, W))
        x = x + self.dp(self.mlp(self.norm2(x)))
        return x


class StochasticDepth(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, device=x.device).bernoulli_(keep) / keep
        return x * mask


# ─────────────────────────────────────────────────────
# Patch Embedding
# ─────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """이미지를 패치로 변환 (overlapping conv)"""
    def __init__(self, in_ch, out_ch, stride=4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 2, 3, stride // 2, 1),
            nn.BatchNorm2d(out_ch // 2),
            nn.GELU(),
            nn.Conv2d(out_ch // 2, out_ch, 3, 2, 1),
            nn.BatchNorm2d(out_ch),
        )
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)   # (B, H*W, C)
        x = self.norm(x)
        return x, H, W


# ─────────────────────────────────────────────────────
# TransNeXt Backbone (4단계 계층)
# ─────────────────────────────────────────────────────

class TransNeXtBackbone(nn.Module):
    """
    4단계 계층적 Transformer Backbone.

    출력 feature pyramid:
        C1: (B, embed[0], H/4,  W/4)
        C2: (B, embed[1], H/8,  W/8)
        C3: (B, embed[2], H/16, W/16)
        C4: (B, embed[3], H/32, W/32)

    Args:
        in_ch:   입력 채널 수 (outdoor: 3ch, fine stage: 4ch)
        embeds:  각 스테이지 채널 수
        depths:  각 스테이지 블록 수
        heads:   각 스테이지 attention head 수
    """

    def __init__(
        self,
        in_ch:  int   = 3,
        embeds: tuple = (64, 128, 256, 512),
        depths: tuple = (2, 2, 6, 2),
        heads:  tuple = (2, 4, 8, 16),
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.embeds = embeds

        # Stochastic depth decay
        total_blocks = sum(depths)
        dp_rates = [x.item() for x in
                    torch.linspace(0, drop_path_rate, total_blocks)]
        cur = 0

        # ── Stage 1 ────────────────────────────────
        self.patch1 = PatchEmbed(in_ch, embeds[0], stride=4)
        self.blocks1 = nn.ModuleList([
            TransNeXtBlock(embeds[0], heads[0], mlp_ratio, drop,
                           dp_rates[cur + i])
            for i in range(depths[0])
        ])
        self.norm1 = nn.LayerNorm(embeds[0])
        cur += depths[0]

        # ── Stage 2 ────────────────────────────────
        self.patch2 = PatchEmbed(embeds[0], embeds[1], stride=2)
        self.blocks2 = nn.ModuleList([
            TransNeXtBlock(embeds[1], heads[1], mlp_ratio, drop,
                           dp_rates[cur + i])
            for i in range(depths[1])
        ])
        self.norm2 = nn.LayerNorm(embeds[1])
        cur += depths[1]

        # ── Stage 3 ────────────────────────────────
        self.patch3 = PatchEmbed(embeds[1], embeds[2], stride=2)
        self.blocks3 = nn.ModuleList([
            TransNeXtBlock(embeds[2], heads[2], mlp_ratio, drop,
                           dp_rates[cur + i])
            for i in range(depths[2])
        ])
        self.norm3 = nn.LayerNorm(embeds[2])
        cur += depths[2]

        # ── Stage 4 ────────────────────────────────
        self.patch4 = PatchEmbed(embeds[2], embeds[3], stride=2)
        self.blocks4 = nn.ModuleList([
            TransNeXtBlock(embeds[3], heads[3], mlp_ratio, drop,
                           dp_rates[cur + i])
            for i in range(depths[3])
        ])
        self.norm4 = nn.LayerNorm(embeds[3])

    def _run_stage(self, x, patch_embed, blocks, norm):
        x, H, W = patch_embed(x)
        for blk in blocks:
            x = blk(x, H, W)
        x = norm(x)
        x = x.transpose(1, 2).reshape(x.shape[0], -1, H, W)
        return x

    def forward(self, x):
        c1 = self._run_stage(x,  self.patch1, self.blocks1, self.norm1)
        c2 = self._run_stage(c1, self.patch2, self.blocks2, self.norm2)
        c3 = self._run_stage(c2, self.patch3, self.blocks3, self.norm3)
        c4 = self._run_stage(c3, self.patch4, self.blocks4, self.norm4)
        return c1, c2, c3, c4   # feature pyramid


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TransNeXtBackbone(in_ch=3).to(device)
    x      = torch.randn(2, 3, 256, 256).to(device)
    c1, c2, c3, c4 = model(x)
    print(f"C1: {c1.shape}")   # (2, 64,  64, 64)
    print(f"C2: {c2.shape}")   # (2, 128, 32, 32)
    print(f"C3: {c3.shape}")   # (2, 256, 16, 16)
    print(f"C4: {c4.shape}")   # (2, 512,  8,  8)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {params:,}")