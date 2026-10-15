#!/usr/bin/env python3
"""
final_qualitative_split_allpairs_v11.py

Generate TWO separate qualitative figures:

1) Error Map Figure
   - RadioUNet-S (35M)
   - TransPathNet-S (38M)
   - RadioUNet-M (97M)
   - TransPathNet-B (111M)

2) Shadow Region Detection Result Figure
   - same 4 models in one figure

Why this version?
-----------------
Instead of mixing Error Map and Shadow Region Detection Result in one single canvas,
this script separates the two visualization types, which makes the figure easier to
read in a paper.

Default qualitative setting
---------------------------
- sample_id: 1486
- frequency: 3750 MHz
- shadow_threshold: -65 dBm

Output
------
Two PNG images:
- qualitative_errormap_allpairs_...
- qualitative_shadow_allpairs_...
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    (
        p
        for p in (SCRIPT_DIR, *SCRIPT_DIR.parents, Path.cwd(), *Path.cwd().parents)
        if (p / "ikeee" / "bratlab_dataset.py").exists()
    ),
    SCRIPT_DIR.parent,
)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "model"))

from ikeee.bratlab_dataset import get_dataloaders

try:
    from ikeee.model.radiounet import build_radiounet
    from ikeee.model.transpathnet_outdoor import build_transpathnet
except Exception:
    from radiounet import build_radiounet
    from transpathnet_outdoor import build_transpathnet


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
MODELS_CONFIG = {
    "radiounet-small": {
        "arch": "radiounet",
        "size": "small",
        "label": "RadioUNet-S (35M)",
    },
    "transpathnet-small": {
        "arch": "transpathnet",
        "size": "small",
        "label": "TransPathNet-S (38M)",
    },
    "radiounet-medium": {
        "arch": "radiounet",
        "size": "medium",
        "label": "RadioUNet-M (97M)",
    },
    "transpathnet-base": {
        "arch": "transpathnet",
        "size": "base",
        "label": "TransPathNet-B (111M)",
    },
}

# Figure order: small-pair on first row, large-pair on second row
MODEL_ORDER = [
    "radiounet-small",
    "transpathnet-small",
    "radiounet-medium",
    "transpathnet-base",
]

COLOR_TP = [0.00, 0.70, 0.15, 0.90]       # green
COLOR_FN = [0.15, 0.25, 1.00, 0.90]       # blue
BUILDING_COLOR = [0.86, 0.86, 0.86, 1.0]  # gray


# ---------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="BRAT-Lab Radiomap dataset root")
    p.add_argument("--ckpt_dir", default="./model_checkpoints")
    p.add_argument("--result_dir", default="./final_results/qualitative_split_allpairs_v11")

    p.add_argument("--freq", type=int, default=3750)
    p.add_argument("--sample_id", type=int, default=1486)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--min_dbm", type=float, default=-90.0)
    p.add_argument("--max_dbm", type=float, default=0.0)
    p.add_argument("--shadow_threshold", type=float, default=-65.0)

    # If error_clip_db > 0: use fixed vmax
    # Else use percentile-based vmax
    p.add_argument("--error_clip_db", type=float, default=0.0)
    p.add_argument("--error_percentile", type=float, default=99.0)

    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def denorm(x, min_dbm: float = -90.0, max_dbm: float = 0.0):
    return x * (max_dbm - min_dbm) + min_dbm


def threshold_tag(threshold: float) -> str:
    if float(threshold).is_integer():
        return str(int(threshold))
    return str(threshold).replace(".", "p").replace("-", "m")


def resolve_ckpt_path(ckpt_dir: str, tag: str, freq: int) -> Path:
    ckpt_root = Path(ckpt_dir)

    # existing special case
    if tag == "radiounet-small" and freq == 4750:
        v2 = ckpt_root / "checkpoints_radiounet_small_4750_v2" / "best.pth"
        if v2.exists():
            return v2

    folder = f"checkpoints_{tag.replace('-', '_')}_{freq}"
    return ckpt_root / folder / "best.pth"


def build_model(tag: str):
    cfg = MODELS_CONFIG[tag]
    if cfg["arch"] == "radiounet":
        return build_radiounet(cfg["size"])
    if cfg["arch"] == "transpathnet":
        return build_transpathnet(cfg["size"])
    raise ValueError(f"Unknown architecture: {cfg['arch']}")


def load_model(tag: str, ckpt_path: Path, device: torch.device):
    model = build_model(tag)
    ckpt = torch.load(str(ckpt_path), map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    clean_state = {}
    for k, v in state.items():
        clean_state[k.replace("module.", "") if k.startswith("module.") else k] = v

    model.load_state_dict(clean_state)
    model.to(device).eval()

    epoch = ckpt.get("epoch", "unknown") if isinstance(ckpt, dict) else "unknown"
    print(f"Loaded {tag:<20} | epoch={epoch} | {ckpt_path}")
    return model


def load_models(ckpt_dir: str, freq: int, device: torch.device) -> Dict[str, torch.nn.Module]:
    models = {}
    for tag in MODEL_ORDER:
        ckpt_path = resolve_ckpt_path(ckpt_dir, tag, freq)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        models[tag] = load_model(tag, ckpt_path, device)
    return models


def find_sample(root: str, freq: int, sample_id: int, img_size: int, num_workers: int):
    _, _, te_loader = get_dataloaders(
        root=root,
        freq=freq,
        img_size=img_size,
        batch_size=1,
        num_workers=num_workers,
    )

    for x, y, n in te_loader:
        cur = int(n[0].item()) if hasattr(n[0], "item") else int(n[0])
        if cur == sample_id:
            return x, y, cur

    raise RuntimeError(f"Sample id {sample_id} not found in test loader for {freq} MHz.")


def predict_all(
    models: Dict[str, torch.nn.Module],
    x: torch.Tensor,
    device: torch.device,
    min_dbm: float,
    max_dbm: float,
) -> Dict[str, np.ndarray]:
    preds = {}
    x_gpu = x.to(device)
    with torch.no_grad():
        for tag, model in models.items():
            _, fine, _ = model(x_gpu)
            pred_dbm = denorm(fine[0, 0].detach().cpu().numpy(), min_dbm, max_dbm)
            preds[tag] = pred_dbm
    return preds


def compute_abs_error(pred_dbm: np.ndarray, gt_dbm: np.ndarray, building: np.ndarray) -> np.ndarray:
    valid = building < 0.5
    err = np.abs(pred_dbm - gt_dbm)
    return np.where(valid, err, np.nan)


def compute_shadow_tp_fn(
    pred_dbm: np.ndarray,
    gt_dbm: np.ndarray,
    building: np.ndarray,
    threshold: float,
):
    valid = building < 0.5
    pred_shadow = (pred_dbm < threshold) & valid
    gt_shadow = (gt_dbm < threshold) & valid

    tp_mask = pred_shadow & gt_shadow
    fn_mask = (~pred_shadow) & gt_shadow

    tp = int(tp_mask.sum())
    fn = int(fn_mask.sum())
    recall = tp / (tp + fn + 1e-8)
    return tp_mask, fn_mask, tp, fn, recall


def get_error_vmax(error_maps: Dict[str, np.ndarray], fixed_clip: float, percentile: float) -> float:
    if fixed_clip and fixed_clip > 0:
        return float(fixed_clip)

    vals = []
    for err in error_maps.values():
        v = err[np.isfinite(err)]
        if v.size:
            vals.append(v)

    if not vals:
        return 10.0

    all_vals = np.concatenate(vals)
    vmax = float(np.percentile(all_vals, percentile))
    return max(1.0, vmax)


def building_overlay(building: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*building.shape, 4), dtype=np.float32)
    rgba[building > 0.5] = BUILDING_COLOR
    return rgba


# ---------------------------------------------------------------------
# Plotting: Error Maps only
# ---------------------------------------------------------------------
def draw_error_panel(fig, ax, model_label: str, error_map: np.ndarray, building: np.ndarray, vmax: float):
    mean_err = float(np.nanmean(error_map))
    max_err = float(np.nanmax(error_map))

    cmap = plt.cm.Reds.copy()
    cmap.set_bad(color="white")

    im = ax.imshow(
        error_map,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        origin="lower",
        interpolation="nearest",
    )
    ax.imshow(building_overlay(building), origin="lower")
    ax.set_title(
        f"{model_label}\nMean: {mean_err:.2f} dB | Max: {max_err:.2f} dB",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax.axis("off")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Absolute RSS Error (dB)", fontsize=10)


def save_errormap_figure(
    preds: Dict[str, np.ndarray],
    gt_dbm: np.ndarray,
    building: np.ndarray,
    freq: int,
    sample_id: int,
    threshold: float,
    fixed_clip: float,
    percentile: float,
    out_path: Path,
    dpi: int,
):
    error_maps = {tag: compute_abs_error(preds[tag], gt_dbm, building) for tag in MODEL_ORDER}
    vmax = get_error_vmax(error_maps, fixed_clip, percentile)

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 11.5))
    fig.subplots_adjust(left=0.05, right=0.96, top=0.88, bottom=0.06, wspace=0.22, hspace=0.25)

    for ax, tag in zip(axes.flat, MODEL_ORDER):
        draw_error_panel(fig, ax, MODELS_CONFIG[tag]["label"], error_maps[tag], building, vmax)

    fig.suptitle(
        f"Qualitative Error Map Comparison\n"
        f"Sample {sample_id} | {freq} MHz | Shadow Threshold {threshold:.1f} dBm",
        fontsize=17,
        fontweight="bold",
        y=0.975,
    )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Error-map vmax used = {vmax:.2f} dB")


# ---------------------------------------------------------------------
# Plotting: Shadow Results only
# ---------------------------------------------------------------------
def draw_shadow_panel(
    ax,
    model_label: str,
    pred_dbm: np.ndarray,
    gt_dbm: np.ndarray,
    building: np.ndarray,
    threshold: float,
):
    tp_mask, fn_mask, tp, fn, recall = compute_shadow_tp_fn(pred_dbm, gt_dbm, building, threshold)

    bg = np.ones((*building.shape, 4), dtype=np.float32)
    ax.imshow(bg, origin="lower")
    ax.imshow(building_overlay(building), origin="lower")

    overlay = np.zeros((*building.shape, 4), dtype=np.float32)
    overlay[tp_mask] = COLOR_TP
    overlay[fn_mask] = COLOR_FN
    ax.imshow(overlay, origin="lower")

    ax.set_title(
        f"{model_label}\nTP: {tp:,} | FN: {fn:,} | Recall: {recall:.3f}",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax.axis("off")


def save_shadow_figure(
    preds: Dict[str, np.ndarray],
    gt_dbm: np.ndarray,
    building: np.ndarray,
    freq: int,
    sample_id: int,
    threshold: float,
    out_path: Path,
    dpi: int,
):
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 11.0))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.10, wspace=0.18, hspace=0.25)

    for ax, tag in zip(axes.flat, MODEL_ORDER):
        draw_shadow_panel(
            ax,
            MODELS_CONFIG[tag]["label"],
            preds[tag],
            gt_dbm,
            building,
            threshold,
        )

    legend_patches = [
        mpatches.Patch(color=COLOR_TP, label="TP: Correctly Detected Shadow"),
        mpatches.Patch(color=COLOR_FN, label="FN: Missed Shadow"),
        mpatches.Patch(color=BUILDING_COLOR, label="Building / Excluded"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=3,
        fontsize=10,
        frameon=True,
    )

    fig.suptitle(
        f"Qualitative Shadow Region Detection Comparison\n"
        f"Sample {sample_id} | {freq} MHz | Shadow Threshold {threshold:.1f} dBm",
        fontsize=17,
        fontweight="bold",
        y=0.975,
    )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    cfg = parse_args()
    out_dir = Path(cfg.result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Fixed sample_id={cfg.sample_id}, freq={cfg.freq} MHz")

    models = load_models(cfg.ckpt_dir, cfg.freq, device)
    x, y, _ = find_sample(cfg.root, cfg.freq, cfg.sample_id, cfg.img_size, cfg.num_workers)

    building = x[0, 0].detach().cpu().numpy()
    gt_dbm = denorm(y[0, 0].detach().cpu().numpy(), cfg.min_dbm, cfg.max_dbm)
    preds = predict_all(models, x, device, cfg.min_dbm, cfg.max_dbm)

    thr_tag = threshold_tag(cfg.shadow_threshold)

    error_out = out_dir / (
        f"qualitative_errormap_allpairs_{cfg.freq}MHz_sample{cfg.sample_id}_th{thr_tag}_v11.png"
    )
    shadow_out = out_dir / (
        f"qualitative_shadow_allpairs_{cfg.freq}MHz_sample{cfg.sample_id}_th{thr_tag}_v11.png"
    )

    save_errormap_figure(
        preds=preds,
        gt_dbm=gt_dbm,
        building=building,
        freq=cfg.freq,
        sample_id=cfg.sample_id,
        threshold=cfg.shadow_threshold,
        fixed_clip=cfg.error_clip_db,
        percentile=cfg.error_percentile,
        out_path=error_out,
        dpi=cfg.dpi,
    )

    save_shadow_figure(
        preds=preds,
        gt_dbm=gt_dbm,
        building=building,
        freq=cfg.freq,
        sample_id=cfg.sample_id,
        threshold=cfg.shadow_threshold,
        out_path=shadow_out,
        dpi=cfg.dpi,
    )

    readme = out_dir / "README_v11.txt"
    readme.write_text(
        "\n".join([
            "[final_qualitative_split_allpairs_v11]",
            f"- fixed sample_id: {cfg.sample_id}",
            f"- fixed frequency: {cfg.freq} MHz",
            "- models shown together:",
            "  * RadioUNet-S (35M)",
            "  * TransPathNet-S (38M)",
            "  * RadioUNet-M (97M)",
            "  * TransPathNet-B (111M)",
            "- output figures are separated by visualization type:",
            "  1) Error Map only",
            "  2) Shadow Region Detection Result only",
            f"- fixed shadow threshold: {cfg.shadow_threshold:.1f} dBm",
            "- error map: absolute RSS error |pred - gt|",
            "- shadow map: GT-shadow-based TP/FN only",
        ]),
        encoding="utf-8",
    )
    print(f"Saved: {readme}")
    print("All done.")


if __name__ == "__main__":
    main()
