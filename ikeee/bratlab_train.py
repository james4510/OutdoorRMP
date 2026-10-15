"""
bratlab_train.py

BRAT-Lab 훈련/평가 통합 스크립트
지원 모델:
    transpathnet-small  (~38.09M, ViT)
    transpathnet-base   (~111.44M, ViT)
    radiounet-small     (~34.92M, CNN)
    radiounet-medium    (~96.98M, CNN)

실행 예시:
    # 훈련
    python bratlab_train.py --mode train \\
        --root /path/to/Radiomap-Data \\
        --freq 4750 --model transpathnet --size small --epochs 100

    # 평가
    python bratlab_train.py --mode eval \\
        --root /path/to/Radiomap-Data \\
        --freq 4750 --model transpathnet --size small \\
        --ckpt ./checkpoints/best.pth
"""

import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import time

from ikeee.bratlab_dataset import get_dataloaders
from ikeee.model.transpathnet_outdoor import build_transpathnet
from ikeee.model.radiounet import build_radiounet

try:
    from pytorch_msssim import ssim as ssim_fn
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False
    print("[Warning] pytorch_msssim 없음 → pip install pytorch-msssim")


# ─────────────────────────────────────────────────────
# 모델 빌더
# ─────────────────────────────────────────────────────

def build_model(model: str, size: str):
    """
    model : 'transpathnet' | 'radiounet'
    size  : 'small' | 'base'  (transpathnet)
             'small' | 'medium' (radiounet)
    """
    if model == "transpathnet":
        assert size in ("small", "base"), \
            "TransPathNet size must be 'small' or 'base'"
        return build_transpathnet(size)

    elif model == "radiounet":
        assert size in ("small", "medium"), \
            "RadioUNet size must be 'small' or 'medium'"
        return build_radiounet(size)

    else:
        raise ValueError(f"Unknown model: {model}")


def model_tag(model: str, size: str):
    """로그/파일명용 태그 반환"""
    return f"{model}-{size}"


def set_seed(seed: int, deterministic: bool = False):
    """Seed Python, NumPy, and PyTorch for repeated-split experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# ─────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────

class RadioLoss(nn.Module):
    def __init__(self, w_mse=0.6, w_ssim=0.3, w_aux=0.1):
        super().__init__()
        self.w_mse  = w_mse
        self.w_ssim = w_ssim
        self.w_aux  = w_aux
        self.mse    = nn.MSELoss()

    def forward(self, pred, aux, target):
        loss = self.w_mse * self.mse(pred, target)
        if HAS_SSIM:
            loss += self.w_ssim * (
                1 - ssim_fn(pred, target, data_range=1.0, size_average=True))
        loss += self.w_aux * self.mse(aux, target)
        return loss


# ─────────────────────────────────────────────────────
# 평가 지표
# ─────────────────────────────────────────────────────

def compute_metrics(pred, target):
    """
    정규화된 [0, 1] radio map 기준 RMSE, MAE, SSIM 계산.
    기존 train/eval 코드 호환용 함수.
    """
    with torch.no_grad():
        pred = pred.float()
        target = target.float()
        rmse = torch.sqrt(((pred - target) ** 2).mean()).item()
        mae  = (pred - target).abs().mean().item()

        if HAS_SSIM:
            ssim = ssim_fn(
                pred, target,
                data_range=1.0,
                size_average=True
            ).item()
        else:
            ssim = 0.0

    return rmse, mae, ssim

def compute_metrics_dbm(pred, target, building_mask):
    """
    논문용 평가 지표.
    - pred, target: 정규화된 [0, 1] tensor, shape (B, 1, H, W)
    - building_mask: 입력 x의 building channel, shape (B, 1, H, W)
    - non-building pixel만 사용
    - RMSE/MAE/Bias는 dBm 단위
    """
    with torch.no_grad():
        pred = pred.float()
        target = target.float()
        building_mask = building_mask.float()
        pred_dbm = denorm(pred)
        gt_dbm   = denorm(target)

        valid = building_mask < 0.5

        diff = pred_dbm[valid] - gt_dbm[valid]

        rmse = torch.sqrt(torch.mean(diff ** 2)).item()
        mae  = torch.mean(torch.abs(diff)).item()
        bias = torch.mean(diff).item()

    return rmse, mae, bias

# ─────────────────────────────────────────────────────
# 훈련
# ─────────────────────────────────────────────────────

def train(cfg):
    set_seed(cfg.seed, cfg.deterministic)
    os.makedirs(cfg.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag    = model_tag(cfg.model, cfg.size)
    print(f"[Train] Device: {device} | Model: {tag} | Seed: {cfg.seed}")

    tr_loader, va_loader, _ = get_dataloaders(
        root=cfg.root, freq=cfg.freq,
        img_size=cfg.img_size, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    model    = build_model(cfg.model, cfg.size).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    amp_enabled = bool(cfg.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    print(f"[Model] {tag} | Params: {n_params/1e6:.2f}M")
    print(
        f"[Train] Batch size: {cfg.batch_size} | "
        f"Accum steps: {cfg.accum_steps} | AMP: {amp_enabled}"
    )

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    criterion = RadioLoss()

    best_loss = float('inf')
    history   = {'train': [], 'val': [], 'rmse': [], 'mae': [], 'ssim': []}

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        # ── Train ──────────────────────────────────
        model.train()
        tr_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, (x, y, _) in enumerate(tr_loader, start=1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                coarse, fine, aux = model(x)
            loss = (criterion(fine.float(), aux.float(), y.float()) +
                    0.4 * criterion(coarse.float(), aux.float(), y.float()))
            tr_loss += loss.item()
            loss_for_backward = loss / cfg.accum_steps
            if amp_enabled:
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            if step % cfg.accum_steps == 0 or step == len(tr_loader):
                if amp_enabled:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        tr_loss /= len(tr_loader)

        # ── Validate ───────────────────────────────
        model.eval()
        va_loss = 0.0
        all_r, all_m, all_s = [], [], []
        with torch.no_grad():
            for x, y, _ in va_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    _, fine, aux = model(x)
                va_loss += criterion(fine.float(), aux.float(), y.float()).item()
                r, m, s = compute_metrics(fine, y)
                all_r.append(r); all_m.append(m); all_s.append(s)
        va_loss /= len(va_loader)
        scheduler.step()

        elapsed = time.time() - t0
        r_mean  = np.mean(all_r)
        m_mean  = np.mean(all_m)
        s_mean  = np.mean(all_s)

        history['train'].append(tr_loss)
        history['val'].append(va_loss)
        history['rmse'].append(r_mean)
        history['mae'].append(m_mean)
        history['ssim'].append(s_mean)

        print(f"[{epoch:03d}/{cfg.epochs}] "
              f"Train:{tr_loss:.4f} Val:{va_loss:.4f} | "
              f"RMSE:{r_mean:.4f} MAE:{m_mean:.4f} SSIM:{s_mean:.4f} | "
              f"{elapsed:.1f}s")

        if va_loss < best_loss:
            best_loss = va_loss
            save_path = os.path.join(cfg.save_dir, 'best.pth')
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'val_loss':    va_loss,
                'model':       cfg.model,
                'size':        cfg.size,
                'config':      vars(cfg),
            }, save_path)
            print(f"  ✅ Best saved → {save_path}")

    _plot_history(history, cfg.save_dir, tag)
    complete_path = os.path.join(cfg.save_dir, 'training_complete.txt')
    with open(complete_path, 'w', encoding='utf-8') as f:
        f.write(
            f"model={tag}\n"
            f"freq={cfg.freq}\n"
            f"seed={cfg.seed}\n"
            f"epochs={cfg.epochs}\n"
            f"batch_size={cfg.batch_size}\n"
            f"accum_steps={cfg.accum_steps}\n"
            f"amp={cfg.amp}\n"
            f"best_val_loss={best_loss}\n"
        )
    print(f"  완료 표시 저장 → {complete_path}")
    print(f"[Train] {tag} 완료!")


def _plot_history(h, save_dir, tag):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(h['train'], label='Train Loss')
    axes[0].plot(h['val'],   label='Val Loss')
    axes[0].set_title(f'{tag} | Loss Curve')
    axes[0].legend(); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')

    axes[1].plot(h['rmse'], label='RMSE')
    axes[1].plot(h['mae'],  label='MAE')
    ax2 = axes[1].twinx()
    ax2.plot(h['ssim'], 'g--', label='SSIM')
    ax2.set_ylabel('SSIM')
    axes[1].set_title(f'{tag} | Metrics')
    axes[1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    axes[1].set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curve.png'),
                dpi=120, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────
# 평가 + 시각화
# ─────────────────────────────────────────────────────

def denorm(t, lo=-90, hi=0):
    return t * (hi - lo) + lo


def evaluate(cfg):
    set_seed(cfg.seed, cfg.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag    = model_tag(cfg.model, cfg.size)

    _, _, te_loader = get_dataloaders(
        root=cfg.root, freq=cfg.freq,
        img_size=cfg.img_size, batch_size=1,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    model = build_model(cfg.model, cfg.size).to(device)
    ckpt  = torch.load(cfg.ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    epoch = ckpt.get('epoch', 'unknown')
    print(f"[Eval] {tag} | Seed: {cfg.seed} | 체크포인트: {cfg.ckpt} (epoch {epoch})")

    os.makedirs(cfg.result_dir, exist_ok=True)

    all_r, all_m, all_s, all_b = [], [], [], []

    with torch.no_grad():
        for i, (x, y, n) in enumerate(te_loader):
            x, y = x.to(device), y.to(device)

            coarse, fine, _ = model(x)

            building = x[:, 0:1]

            r, m, b = compute_metrics_dbm(fine, y, building)

            if HAS_SSIM:
                s = ssim_fn(fine, y, data_range=1.0, size_average=True).item()
            else:
                s = 0.0

            all_r.append(r)
            all_m.append(m)
            all_b.append(b)
            all_s.append(s)

            if i < cfg.n_vis:
                _visualize(
                    x.cpu(), coarse.cpu(), fine.cpu(),
                    y.cpu(), n[0].item(), i,
                    cfg.result_dir, r, m, s, tag
                )

    print(f"\n{'='*45}")
    print(f"  Model      : {tag}")
    print(f"  Freq       : {cfg.freq} MHz")
    print(f"  Test RMSE  : {np.mean(all_r):.4f} dBm")
    print(f"  Test MAE   : {np.mean(all_m):.4f} dBm")
    print(f"  Test Bias  : {np.mean(all_b):+.4f} dB")
    print(f"  Test SSIM  : {np.mean(all_s):.4f}")
    print(f"{'='*45}")


def _visualize(x, coarse, fine, gt, sample_n, idx,
               save_dir, rmse, mae, ssim, tag):
    building = x[0, 0].numpy()
    gt_dbm   = denorm(gt[0, 0].numpy())
    co_dbm   = denorm(coarse[0, 0].numpy())
    fi_dbm   = denorm(fine[0, 0].numpy())
    resid    = fi_dbm - gt_dbm

    mask   = building > 0.5
    cmap   = plt.cm.jet.copy(); cmap.set_bad('white')
    gt_m   = np.ma.masked_where(mask, gt_dbm)
    co_m   = np.ma.masked_where(mask, co_dbm)
    fi_m   = np.ma.masked_where(mask, fi_dbm)
    re_m   = np.ma.masked_where(mask, resid)

    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    panels = [
        (building, 'gray',     'Input: Building Map', {}),
        (gt_m,     cmap,       'GT Radio Map',        {'vmin': -90, 'vmax': 0}),
        (co_m,     cmap,       'Coarse Prediction',   {'vmin': -90, 'vmax': 0}),
        (fi_m,     cmap,       'Fine Prediction',     {'vmin': -90, 'vmax': 0}),
        (re_m,     'coolwarm', 'Residual (Fine-GT)',  {'vmin': -20, 'vmax': 20}),
    ]
    for ax, (d, cm, t, kw) in zip(axes, panels):
        im = ax.imshow(d, cmap=cm, origin='upper', **kw)
        ax.set_title(t, fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')

    mean_res = np.abs(re_m.compressed()).mean() if re_m.compressed().size else 0
    axes[-1].set_xlabel(f'Mean |Residual|: {mean_res:.1f} dBm', fontsize=9)

    fig.suptitle(
        f'{tag} | Sample #{sample_n} | '
        f'RMSE:{rmse:.3f}  MAE:{mae:.3f}  SSIM:{ssim:.3f}',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    path = os.path.join(save_dir, f'result_{idx+1:03d}_sample{sample_n}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  저장: {path}")


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",        default="train",
                        choices=["train", "eval"])
    # 데이터
    parser.add_argument("--root",        required=True,
                        help="Radiomap-Data 폴더 경로")
    parser.add_argument("--freq",        type=int, default=4750,
                        choices=[1750, 2750, 3750, 4750, 5750])
    # 모델
    parser.add_argument("--model",       default="transpathnet",
                        choices=["transpathnet", "radiounet"])
    parser.add_argument("--size",        default="small",
                        choices=["small", "medium", "base"],
                        help="transpathnet: small|base / radiounet: small|medium")
    parser.add_argument("--img_size",    type=int, default=256)
    # 훈련
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--accum_steps", type=int,   default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--amp",         action="store_true",
                        help="Use CUDA automatic mixed precision during training")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--save_dir",    default="./checkpoints")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed for split generation and training")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic CuDNN settings for repeatability")
    # 평가
    parser.add_argument("--ckpt",        default="./checkpoints/best.pth")
    parser.add_argument("--result_dir",  default="./results")
    parser.add_argument("--n_vis",       type=int, default=10)
    cfg = parser.parse_args()

    if cfg.mode == "train":
        train(cfg)
    else:
        evaluate(cfg)
