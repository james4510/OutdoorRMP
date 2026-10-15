#!/usr/bin/env python3
"""
final_eval_metrics_fixed_threshold_heatmap.py

Fixed-threshold evaluation script for shadow-region detection.

Purpose
-------
Recalculate Precision, Recall, IoU, and F1-score under fixed RSS thresholds
(-70, -65, -60 dBm by default), and generate paper-ready heatmaps and tables.

This script replaces the old frequency-dependent bottom-25% threshold evaluation
when the paper uses fixed absolute RSS thresholds.

Main outputs
------------
1) per_sample_metrics_fixed_threshold.csv
   - sample-wise TP/FP/FN/TN, Precision, Recall, IoU, F1, RMSE, etc.

2) summary_metrics_fixed_threshold.csv
   - mean ± std across test samples for each frequency / model / threshold.

3) heatmap_fixed_threshold_th{threshold}.png
   - 2 x 2 heatmap for Precision / Recall / IoU / F1-score.

4) table_fixed_threshold_th{threshold}.png
   - summary table image with mean ± std.

Recommended usage
-----------------
python final_eval_metrics_fixed_threshold_heatmap.py \
    --root /home/james4510/OutdoorRadioMapPrediction/data \
    --ckpt_dir ./model_checkpoints \
    --result_dir ./final_results/fixed_threshold_metrics_heatmap \
    --thresholds -70 -65 -60 \
    --freqs 1750 2750 3750 4750 5750 \
    --num_workers 4

If you want a paper figure centered on the representative threshold only:
python final_eval_metrics_fixed_threshold_heatmap.py \
    --root /home/james4510/OutdoorRadioMapPrediction/data \
    --ckpt_dir ./model_checkpoints \
    --result_dir ./final_results/fixed_threshold_metrics_heatmap \
    --thresholds -65
"""

from __future__ import annotations

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
import matplotlib.pyplot as plt


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


def _import_get_dataloaders():
    import ikeee.bratlab_dataset as ds_mod

    # Prefer explicitly fixed dataloader if it exists.
    if hasattr(ds_mod, "get_dataloaders_fixed"):
        print("[Import] Using get_dataloaders_fixed")
        return ds_mod.get_dataloaders_fixed
    if hasattr(ds_mod, "get_dataloaders_fized"):
        print("[Import] Using get_dataloaders_fized")
        return ds_mod.get_dataloaders_fized

    # In the current bratlab_dataset.py, get_dataloaders already uses
    # independent train/val/test datasets and disables augmentation for test.
    print("[Import] Using get_dataloaders")
    return ds_mod.get_dataloaders


get_dataloaders = _import_get_dataloaders()

try:
    from ikeee.model.radiounet import build_radiounet
    from ikeee.model.transpathnet_outdoor import build_transpathnet
except Exception:
    from radiounet import build_radiounet
    from transpathnet_outdoor import build_transpathnet


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DEFAULT_FREQS = [1750, 2750, 3750, 4750, 5750]
DEFAULT_THRESHOLDS = [-70.0, -65.0, -60.0]

MODELS_CONFIG = {
    "radiounet-small": {
        "arch": "radiounet",
        "size": "small",
        "label": "RadioUNet-S\n(35M)",
        "table_label": "RadioUNet-S (35M)",
    },
    "radiounet-medium": {
        "arch": "radiounet",
        "size": "medium",
        "label": "RadioUNet-M\n(97M)",
        "table_label": "RadioUNet-M (97M)",
    },
    "transpathnet-small": {
        "arch": "transpathnet",
        "size": "small",
        "label": "TransPathNet-S\n(38M)",
        "table_label": "TransPathNet-S (38M)",
    },
    "transpathnet-base": {
        "arch": "transpathnet",
        "size": "base",
        "label": "TransPathNet-B\n(111M)",
        "table_label": "TransPathNet-B (111M)",
    },
}

MODEL_ORDER = [
    "radiounet-small",
    "radiounet-medium",
    "transpathnet-small",
    "transpathnet-base",
]

METRICS = ["precision", "recall", "iou", "f1"]
METRIC_TITLES = {
    "precision": "Precision",
    "recall": "Recall",
    "iou": "IoU",
    "f1": "F1-score",
}


# ---------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--root", required=True, help="BRAT-Lab Radiomap dataset root")
    p.add_argument("--ckpt_dir", default="./model_checkpoints")
    p.add_argument("--result_dir", default="./final_results/fixed_threshold_metrics_heatmap")

    p.add_argument("--freqs", type=int, nargs="+", default=DEFAULT_FREQS)
    p.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    p.add_argument("--models", nargs="+", default=MODEL_ORDER, choices=MODEL_ORDER)

    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed used for train/val/test split generation")

    p.add_argument("--min_dbm", type=float, default=-90.0)
    p.add_argument("--max_dbm", type=float, default=0.0)

    p.add_argument("--heatmap_vmin", type=float, default=0.0)
    p.add_argument("--heatmap_vmax", type=float, default=1.0)
    p.add_argument("--cmap", default="viridis")
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

    # User requested the v2 checkpoint for RadioUNet-S at 4750 MHz.
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

    # Remove DataParallel prefix if present.
    clean_state = {}
    for k, v in state.items():
        clean_state[k.replace("module.", "") if k.startswith("module.") else k] = v

    model.load_state_dict(clean_state)
    model.to(device).eval()

    epoch = ckpt.get("epoch", "unknown") if isinstance(ckpt, dict) else "unknown"
    print(f"  Loaded {tag:<20} | epoch={epoch} | {ckpt_path}")
    return model


def load_models_for_freq(ckpt_dir: str, freq: int, model_tags: List[str],
                         device: torch.device) -> Dict[str, torch.nn.Module]:
    models = {}
    for tag in model_tags:
        ckpt_path = resolve_ckpt_path(ckpt_dir, tag, freq)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        models[tag] = load_model(tag, ckpt_path, device)
    return models


def get_fine_output(model_output):
    """
    Existing models usually return (_, fine, _).
    This helper also supports a model that directly returns a tensor.
    """
    if isinstance(model_output, (tuple, list)):
        if len(model_output) >= 2:
            return model_output[1]
        return model_output[0]
    return model_output


def sample_id_from_batch(n, batch_index: int) -> int:
    if isinstance(n, torch.Tensor):
        if n.ndim == 0:
            return int(n.item())
        return int(n[batch_index].item())
    if isinstance(n, (list, tuple)):
        item = n[batch_index]
        if isinstance(item, torch.Tensor):
            return int(item.item())
        return int(item)
    return int(n)


def compute_shadow_metrics(
    pred_dbm: np.ndarray,
    gt_dbm: np.ndarray,
    building_mask: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    valid = building_mask < 0.5

    pred_shadow = (pred_dbm < threshold) & valid
    gt_shadow = (gt_dbm < threshold) & valid

    TP = int((pred_shadow & gt_shadow).sum())
    FP = int((pred_shadow & ~gt_shadow).sum())
    FN = int((~pred_shadow & gt_shadow).sum())
    TN = int((~pred_shadow & ~gt_shadow & valid).sum())

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    miss_rate = FN / (TP + FN + 1e-8)
    false_alarm_rate = FP / (FP + TN + 1e-8)

    gt_shadow_pixels = int(gt_shadow.sum())
    pred_shadow_pixels = int(pred_shadow.sum())
    valid_pixels = int(valid.sum())

    # RMSE is independent of the shadow threshold, but useful for consistency.
    if valid_pixels > 0:
        diff = pred_dbm[valid] - gt_dbm[valid]
        rmse_dbm = float(np.sqrt(np.mean(diff ** 2)))
        mae_dbm = float(np.mean(np.abs(diff)))
    else:
        rmse_dbm = np.nan
        mae_dbm = np.nan

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "gt_shadow_pixels": gt_shadow_pixels,
        "pred_shadow_pixels": pred_shadow_pixels,
        "valid_pixels": valid_pixels,
        "gt_shadow_ratio": gt_shadow_pixels / (valid_pixels + 1e-8),
        "pred_shadow_ratio": pred_shadow_pixels / (valid_pixels + 1e-8),
        "precision": float(precision),
        "recall": float(recall),
        "iou": float(iou),
        "f1": float(f1),
        "miss_rate": float(miss_rate),
        "false_alarm_rate": float(false_alarm_rate),
        "rmse_dbm": rmse_dbm,
        "mae_dbm": mae_dbm,
    }


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
def evaluate(cfg: argparse.Namespace) -> List[Dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Split seed: {cfg.seed}")

    rows: List[Dict[str, Any]] = []

    for freq in cfg.freqs:
        print("\n" + "=" * 80)
        print(f"Frequency: {freq} MHz")
        print("=" * 80)

        models = load_models_for_freq(cfg.ckpt_dir, freq, cfg.models, device)

        _, _, te_loader = get_dataloaders(
            root=cfg.root,
            freq=freq,
            img_size=cfg.img_size,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            seed=cfg.seed,
        )

        with torch.no_grad():
            for batch_idx, (x, y, n) in enumerate(te_loader):
                x = x.to(device)
                y = y.to(device)

                # Predictions for all models for this batch.
                pred_by_model = {}
                for tag, model in models.items():
                    out = model(x)
                    fine = get_fine_output(out)
                    pred_by_model[tag] = fine.detach().cpu().numpy()

                x_np = x.detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()

                bsz = x_np.shape[0]
                for b in range(bsz):
                    sample_id = sample_id_from_batch(n, b)

                    building = x_np[b, 0]
                    gt_norm = y_np[b, 0]
                    gt_dbm = denorm(gt_norm, cfg.min_dbm, cfg.max_dbm)

                    for tag in cfg.models:
                        pred_norm = pred_by_model[tag][b, 0]
                        pred_dbm = denorm(pred_norm, cfg.min_dbm, cfg.max_dbm)

                        for threshold in cfg.thresholds:
                            m = compute_shadow_metrics(
                                pred_dbm=pred_dbm,
                                gt_dbm=gt_dbm,
                                building_mask=building,
                                threshold=threshold,
                            )

                            row = {
                                "seed": int(cfg.seed),
                                "freq_mhz": int(freq),
                                "sample_id": int(sample_id),
                                "sample_rank_in_test_loader": int(batch_idx * bsz + b),
                                "model_tag": tag,
                                "model_label": MODELS_CONFIG[tag]["table_label"],
                                "threshold_dbm": float(threshold),
                            }
                            row.update(m)
                            rows.append(row)

                if (batch_idx + 1) % 50 == 0:
                    print(f"  {batch_idx + 1}/{len(te_loader)} batches done")

    return rows


# ---------------------------------------------------------------------
# CSV summary
# ---------------------------------------------------------------------
def save_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, Any]], freqs: List[int],
              model_tags: List[str], thresholds: List[float]) -> List[Dict[str, Any]]:
    summary = []

    for threshold in thresholds:
        for freq in freqs:
            for tag in model_tags:
                sub = [
                    r for r in rows
                    if int(r["freq_mhz"]) == int(freq)
                    and r["model_tag"] == tag
                    and abs(float(r["threshold_dbm"]) - float(threshold)) < 1e-6
                ]

                if not sub:
                    continue

                item = {
                    "seed": int(sub[0].get("seed", -1)),
                    "threshold_dbm": float(threshold),
                    "freq_mhz": int(freq),
                    "model_tag": tag,
                    "model_label": MODELS_CONFIG[tag]["table_label"],
                    "n_samples": len(sub),
                }

                # Mean/std of sample-wise metrics.
                for metric in [
                    "precision", "recall", "iou", "f1",
                    "miss_rate", "false_alarm_rate",
                    "rmse_dbm", "mae_dbm",
                    "gt_shadow_ratio", "pred_shadow_ratio",
                ]:
                    vals = np.asarray([float(r[metric]) for r in sub], dtype=float)
                    vals = vals[np.isfinite(vals)]
                    item[f"{metric}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
                    item[f"{metric}_std"] = float(np.std(vals)) if len(vals) else np.nan

                # Micro metrics from accumulated counts.
                TP = int(sum(int(r["TP"]) for r in sub))
                FP = int(sum(int(r["FP"]) for r in sub))
                FN = int(sum(int(r["FN"]) for r in sub))
                TN = int(sum(int(r["TN"]) for r in sub))
                item["micro_TP"] = TP
                item["micro_FP"] = FP
                item["micro_FN"] = FN
                item["micro_TN"] = TN
                item["micro_precision"] = TP / (TP + FP + 1e-8)
                item["micro_recall"] = TP / (TP + FN + 1e-8)
                item["micro_iou"] = TP / (TP + FP + FN + 1e-8)
                p = item["micro_precision"]
                r = item["micro_recall"]
                item["micro_f1"] = 2 * p * r / (p + r + 1e-8)

                summary.append(item)

    return summary


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def value_from_summary(summary: List[Dict[str, Any]], threshold: float, freq: int,
                       model_tag: str, metric: str) -> float:
    for r in summary:
        if (
            abs(float(r["threshold_dbm"]) - float(threshold)) < 1e-6
            and int(r["freq_mhz"]) == int(freq)
            and r["model_tag"] == model_tag
        ):
            return float(r[f"{metric}_mean"])
    return np.nan


def std_from_summary(summary: List[Dict[str, Any]], threshold: float, freq: int,
                     model_tag: str, metric: str) -> float:
    for r in summary:
        if (
            abs(float(r["threshold_dbm"]) - float(threshold)) < 1e-6
            and int(r["freq_mhz"]) == int(freq)
            and r["model_tag"] == model_tag
        ):
            return float(r[f"{metric}_std"])
    return np.nan


def plot_heatmap_for_threshold(
    summary: List[Dict[str, Any]],
    threshold: float,
    freqs: List[int],
    model_tags: List[str],
    out_dir: Path,
    vmin: float,
    vmax: float,
    cmap_name: str,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    axes = axes.flatten()

    for ax, metric in zip(axes, METRICS):
        mat = np.zeros((len(freqs), len(model_tags)), dtype=float)
        for i, freq in enumerate(freqs):
            for j, tag in enumerate(model_tags):
                mat[i, j] = value_from_summary(summary, threshold, freq, tag, metric)

        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap_name, aspect="auto")

        ax.set_title(METRIC_TITLES[metric], fontsize=15, fontweight="bold")
        ax.set_xticks(np.arange(len(model_tags)))
        ax.set_xticklabels([MODELS_CONFIG[t]["label"] for t in model_tags], fontsize=20)
        ax.set_yticks(np.arange(len(freqs)))
        ax.set_yticklabels([f"{f}" for f in freqs], fontsize=20)
        ax.set_xlabel("Model", fontsize=21)
        ax.set_ylabel("Frequency (MHz)", fontsize=21)

        # Annotate values and bold the best model for each frequency.
        for i in range(len(freqs)):
            row_vals = mat[i, :]
            best_j = int(np.nanargmax(row_vals)) if np.isfinite(row_vals).any() else -1
            for j in range(len(model_tags)):
                val = mat[i, j]
                txt = "N/A" if not np.isfinite(val) else f"{val:.3f}"
                weight = "bold" if j == best_j else "normal"
                # Choose readable text color.
                text_color = "white" if np.isfinite(val) and val < (vmin + vmax) / 2 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=20, fontweight=weight, color=text_color)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Score", fontsize=20)

    fig.suptitle(
        f"Shadow Detection Metrics under Fixed Threshold {threshold:.1f} dBm",
        fontsize=18,
        fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    th_tag = threshold_tag(threshold)
    path = out_dir / f"heatmap_fixed_threshold_th{th_tag}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_table_for_threshold(
    summary: List[Dict[str, Any]],
    threshold: float,
    freqs: List[int],
    model_tags: List[str],
    out_dir: Path,
    dpi: int,
) -> None:
    columns = ["Freq.", "Model", "Precision", "Recall", "IoU", "F1-score"]
    cell_rows = []

    for freq in freqs:
        for tag in model_tags:
            row = [str(freq), MODELS_CONFIG[tag]["table_label"]]
            for metric in METRICS:
                mean = value_from_summary(summary, threshold, freq, tag, metric)
                std = std_from_summary(summary, threshold, freq, tag, metric)
                if np.isfinite(mean) and np.isfinite(std):
                    row.append(f"{mean:.3f}±{std:.3f}")
                else:
                    row.append("N/A")
            cell_rows.append(row)

    fig_h = max(7.0, 0.38 * len(cell_rows) + 2.0)
    fig, ax = plt.subplots(figsize=(13.5, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=cell_rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.35)

    # Header style.
    for col in range(len(columns)):
        table[(0, col)].set_text_props(weight="bold")
        table[(0, col)].set_facecolor("#e6e6e6")

    # Lightly separate frequency groups.
    for i in range(1, len(cell_rows) + 1):
        if (i - 1) % len(model_tags) == 0:
            for col in range(len(columns)):
                table[(i, col)].set_linewidth(1.3)

    ax.set_title(
        f"Fixed-threshold Shadow Detection Metrics (mean ± std) | Threshold {threshold:.1f} dBm",
        fontsize=16,
        fontweight="bold",
        pad=16,
    )

    th_tag = threshold_tag(threshold)
    path = out_dir / f"table_fixed_threshold_th{th_tag}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def print_summary_to_console(summary: List[Dict[str, Any]], thresholds: List[float],
                             freqs: List[int], model_tags: List[str]) -> None:
    for threshold in thresholds:
        print("\n" + "=" * 100)
        print(f"Fixed threshold = {threshold:.1f} dBm | sample-wise mean")
        print("=" * 100)
        print(f"{'Freq':>6}  {'Model':<24}  {'Prec':>7}  {'Recall':>7}  {'IoU':>7}  {'F1':>7}")
        print("-" * 100)
        for freq in freqs:
            for tag in model_tags:
                vals = []
                for metric in METRICS:
                    vals.append(value_from_summary(summary, threshold, freq, tag, metric))
                print(
                    f"{freq:>6}  {MODELS_CONFIG[tag]['table_label']:<24}  "
                    f"{vals[0]:>7.3f}  {vals[1]:>7.3f}  {vals[2]:>7.3f}  {vals[3]:>7.3f}"
                )
            print("-" * 100)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = evaluate(cfg)

    per_sample_fields = [
        "seed", "freq_mhz", "sample_id", "sample_rank_in_test_loader",
        "model_tag", "model_label", "threshold_dbm",
        "TP", "FP", "FN", "TN",
        "gt_shadow_pixels", "pred_shadow_pixels", "valid_pixels",
        "gt_shadow_ratio", "pred_shadow_ratio",
        "precision", "recall", "iou", "f1",
        "miss_rate", "false_alarm_rate",
        "rmse_dbm", "mae_dbm",
    ]
    per_sample_path = out_dir / "per_sample_metrics_fixed_threshold.csv"
    save_csv(per_sample_path, rows, per_sample_fields)
    print(f"Saved: {per_sample_path}")

    summary = summarize(rows, cfg.freqs, cfg.models, cfg.thresholds)

    summary_fields = [
        "seed", "threshold_dbm", "freq_mhz", "model_tag", "model_label", "n_samples",
    ]
    for metric in [
        "precision", "recall", "iou", "f1",
        "miss_rate", "false_alarm_rate",
        "rmse_dbm", "mae_dbm",
        "gt_shadow_ratio", "pred_shadow_ratio",
    ]:
        summary_fields += [f"{metric}_mean", f"{metric}_std"]
    summary_fields += [
        "micro_TP", "micro_FP", "micro_FN", "micro_TN",
        "micro_precision", "micro_recall", "micro_iou", "micro_f1",
    ]

    summary_path = out_dir / "summary_metrics_fixed_threshold.csv"
    save_csv(summary_path, summary, summary_fields)
    print(f"Saved: {summary_path}")

    print_summary_to_console(summary, cfg.thresholds, cfg.freqs, cfg.models)

    for threshold in cfg.thresholds:
        plot_heatmap_for_threshold(
            summary=summary,
            threshold=threshold,
            freqs=cfg.freqs,
            model_tags=cfg.models,
            out_dir=out_dir,
            vmin=cfg.heatmap_vmin,
            vmax=cfg.heatmap_vmax,
            cmap_name=cfg.cmap,
            dpi=cfg.dpi,
        )
        plot_table_for_threshold(
            summary=summary,
            threshold=threshold,
            freqs=cfg.freqs,
            model_tags=cfg.models,
            out_dir=out_dir,
            dpi=cfg.dpi,
        )

    readme = out_dir / "README_fixed_threshold_metrics.txt"
    readme.write_text(
        "\n".join([
            "[final_eval_metrics_fixed_threshold_heatmap]",
            "- Shadow labels are defined using fixed absolute RSS thresholds.",
            f"- thresholds: {cfg.thresholds}",
            "- pred_shadow = pred_RSS < threshold",
            "- gt_shadow   = gt_RSS   < threshold",
            "- building / excluded pixels are removed from evaluation.",
            "- heatmaps use sample-wise mean Precision / Recall / IoU / F1-score.",
            "- summary CSV also includes micro metrics from accumulated TP/FP/FN/TN.",
            "- RMSE and MAE are threshold-independent but saved for consistency.",
            f"- split seed: {cfg.seed}",
        ]),
        encoding="utf-8",
    )
    print(f"Saved: {readme}")
    print("All done.")


if __name__ == "__main__":
    main()
