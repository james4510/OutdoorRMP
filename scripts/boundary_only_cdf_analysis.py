#!/usr/bin/env python3
"""
boundary_only_cdf_analysis.py

Boundary-only CDF analysis for Radio Map Estimation shadow detection.

Why boundary-only?
------------------
A full-image Miss Rate CDF can move left at higher frequencies because a fixed
threshold may create a wider GT shadow region. To directly test whether sharper
shadow boundaries are harder to predict, this script evaluates only boundary
pixels, defined by:

    boundary_mask = valid_pixels AND |GT_RSS - threshold| <= boundary_width_dbm

Default boundary width is ±10 dB, matching the paper draft idea.

Metrics
-------
For each test sample, model, frequency, and threshold:

1) Boundary-only error
   - boundary_mae_db  = mean(|Pred - GT|) over boundary pixels
   - boundary_rmse_db = sqrt(mean((Pred - GT)^2)) over boundary pixels

2) Boundary-only Miss Rate
   - boundary_miss_rate = boundary_FN / boundary_GT_shadow_pixels
   - computed only within boundary pixels
   - boundary_GT_shadow_pixels = boundary pixels whose GT RSS < threshold

Main outputs
------------
- per_sample_boundary_metrics.csv
- boundary_cdf_summary.csv
- fig_boundary_error_cdf_th{T}_bw{W}.png
- fig_boundary_missrate_cdf_th{T}_bw{W}.png
- fig_boundary_combined_cdf_th{T}_bw{W}.png

Recommended use
---------------
python boundary_only_cdf_analysis.py \
    --root /home/james4510/OutdoorRadioMapPrediction/data \
    --ckpt_dir ./model_checkpoints \
    --result_dir ./final_results/boundary_cdf \
    --freqs 1750 3750 5750 \
    --threshold_mode fixed \
    --thresholds -65 \
    --boundary_width_dbm 10

For threshold sensitivity:
python boundary_only_cdf_analysis.py \
    --root /home/james4510/OutdoorRadioMapPrediction/data \
    --ckpt_dir ./model_checkpoints \
    --result_dir ./final_results/boundary_cdf \
    --freqs 1750 3750 5750 \
    --threshold_mode fixed \
    --thresholds -70 -65 -60 \
    --boundary_width_dbm 10

For the original per-frequency 25% thresholds:
python boundary_only_cdf_analysis.py \
    --root /home/james4510/OutdoorRadioMapPrediction/data \
    --ckpt_dir ./model_checkpoints \
    --result_dir ./final_results/boundary_cdf_perfreq \
    --freqs 1750 3750 5750 \
    --threshold_mode per_freq \
    --boundary_width_dbm 10
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

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
FREQ_THRESHOLD = {
    1750: -61.7,
    2750: -64.9,
    3750: -67.4,
    4750: -69.5,
    5750: -71.1,
}

ALL_FREQS = [1750, 2750, 3750, 4750, 5750]

MODELS_CONFIG = {
    "radiounet-small":    {"arch": "radiounet",    "size": "small",  "label": "RadioUNet-S (35M)"},
    "radiounet-medium":   {"arch": "radiounet",    "size": "medium", "label": "RadioUNet-M (97M)"},
    "transpathnet-small": {"arch": "transpathnet", "size": "small",  "label": "TransPathNet-S (38M)"},
    "transpathnet-base":  {"arch": "transpathnet", "size": "base",   "label": "TransPathNet-B (111M)"},
}

MODEL_ORDER = [
    "radiounet-small",
    "radiounet-medium",
    "transpathnet-small",
    "transpathnet-base",
]

MODEL_STYLE = {
    "radiounet-small":    {"color": "tab:blue",   "linestyle": "-",  "linewidth": 2.3},
    "radiounet-medium":   {"color": "tab:orange", "linestyle": "-",  "linewidth": 2.3},
    "transpathnet-small": {"color": "tab:green",  "linestyle": "--", "linewidth": 2.3},
    "transpathnet-base":  {"color": "tab:red",    "linestyle": "--", "linewidth": 2.3},
}


# ---------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="BRAT-Lab Radiomap dataset root")
    p.add_argument("--ckpt_dir", default="./model_checkpoints")
    p.add_argument("--result_dir", default="./final_results/boundary_cdf")

    p.add_argument("--freqs", type=int, nargs="+", default=[1750, 3750, 5750])
    p.add_argument("--models", nargs="+", default=MODEL_ORDER, choices=MODEL_ORDER)

    p.add_argument("--threshold_mode", choices=["fixed", "per_freq"], default="fixed")
    p.add_argument("--thresholds", type=float, nargs="+", default=[-65.0],
                   help="Used only when threshold_mode=fixed")
    p.add_argument("--threshold_tol", type=float, default=1e-6)

    p.add_argument("--boundary_width_dbm", type=float, default=10.0,
                   help="Boundary pixels satisfy |GT RSS - threshold| <= this value")

    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--min_dbm", type=float, default=-90.0)
    p.add_argument("--max_dbm", type=float, default=0.0)

    p.add_argument("--error_metric", choices=["boundary_mae_db", "boundary_rmse_db"],
                   default="boundary_mae_db")
    p.add_argument("--error_x_max", type=float, default=12.0)
    p.add_argument("--error_x_step", type=float, default=0.05)
    p.add_argument("--miss_x_max", type=float, default=1.0)
    p.add_argument("--miss_x_step", type=float, default=0.0025)

    p.add_argument("--summary_error_cutoffs", type=float, nargs="+", default=[1.0, 2.0, 5.0])
    p.add_argument("--summary_miss_cutoffs", type=float, nargs="+", default=[0.05, 0.10, 0.20])

    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def denorm(x: np.ndarray, min_dbm: float = -90.0, max_dbm: float = 0.0) -> np.ndarray:
    return x * (max_dbm - min_dbm) + min_dbm


def threshold_tag(threshold: Optional[float], mode: str) -> str:
    if mode == "per_freq":
        return "perfreq"
    assert threshold is not None
    if float(threshold).is_integer():
        return str(int(threshold))
    return str(threshold).replace(".", "p").replace("-", "m")


def width_tag(width: float) -> str:
    if float(width).is_integer():
        return str(int(width))
    return str(width).replace(".", "p")


def get_thresholds_for_freq(freq: int, cfg: argparse.Namespace) -> List[float]:
    if cfg.threshold_mode == "per_freq":
        return [float(FREQ_THRESHOLD[freq])]
    return [float(t) for t in cfg.thresholds]


def resolve_ckpt_path(ckpt_dir: str, tag: str, freq: int) -> Path:
    ckpt_root = Path(ckpt_dir)

    # Keep existing special case for RadioUNet-S at 4750 MHz.
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
    print(f"  Loaded {tag:<20} | epoch={epoch} | {ckpt_path}")
    return model


def load_models_for_freq(ckpt_dir: str, freq: int, model_tags: List[str],
                         device: torch.device) -> Dict[str, torch.nn.Module]:
    models = {}
    for tag in model_tags:
        ckpt_path = resolve_ckpt_path(ckpt_dir, tag, freq)
        if not ckpt_path.exists():
            print(f"  [SKIP] Missing checkpoint: {ckpt_path}")
            continue
        models[tag] = load_model(tag, ckpt_path, device)
    return models


def empirical_cdf(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.zeros_like(x_grid)
    values = np.sort(values)
    return np.searchsorted(values, x_grid, side="right") / len(values)


# ---------------------------------------------------------------------
# Boundary metrics
# ---------------------------------------------------------------------
def compute_boundary_metrics(pred_dbm: np.ndarray,
                             gt_dbm: np.ndarray,
                             building: np.ndarray,
                             threshold: float,
                             boundary_width_dbm: float) -> Dict[str, object]:
    valid = building < 0.5
    boundary = valid & (np.abs(gt_dbm - threshold) <= boundary_width_dbm)

    pred_shadow = (pred_dbm < threshold) & boundary
    gt_shadow = (gt_dbm < threshold) & boundary
    gt_nonshadow = boundary & ~gt_shadow

    TP = int((pred_shadow & gt_shadow).sum())
    FP = int((pred_shadow & ~gt_shadow).sum())
    FN = int((~pred_shadow & gt_shadow).sum())
    TN = int((~pred_shadow & ~gt_shadow & boundary).sum())

    boundary_pixels = int(boundary.sum())
    gt_shadow_pixels = int(gt_shadow.sum())
    gt_nonshadow_pixels = int(gt_nonshadow.sum())

    if boundary_pixels > 0:
        err = pred_dbm[boundary] - gt_dbm[boundary]
        abs_err = np.abs(err)
        boundary_mae_db = float(np.mean(abs_err))
        boundary_rmse_db = float(np.sqrt(np.mean(err ** 2)))
        boundary_bias_db = float(np.mean(err))
        boundary_abs_p90_db = float(np.percentile(abs_err, 90))
    else:
        boundary_mae_db = np.nan
        boundary_rmse_db = np.nan
        boundary_bias_db = np.nan
        boundary_abs_p90_db = np.nan

    boundary_miss_rate = np.nan if gt_shadow_pixels == 0 else FN / gt_shadow_pixels
    boundary_false_alarm_rate = np.nan if gt_nonshadow_pixels == 0 else FP / gt_nonshadow_pixels

    boundary_precision = TP / (TP + FP + 1e-8)
    boundary_recall = TP / (TP + FN + 1e-8)
    boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall + 1e-8)
    boundary_iou = TP / (TP + FP + FN + 1e-8)

    return {
        "boundary_pixels": boundary_pixels,
        "boundary_gt_shadow_pixels": gt_shadow_pixels,
        "boundary_gt_nonshadow_pixels": gt_nonshadow_pixels,
        "boundary_TP": TP,
        "boundary_FP": FP,
        "boundary_FN": FN,
        "boundary_TN": TN,
        "boundary_mae_db": boundary_mae_db,
        "boundary_rmse_db": boundary_rmse_db,
        "boundary_bias_db": boundary_bias_db,
        "boundary_abs_p90_db": boundary_abs_p90_db,
        "boundary_miss_rate": float(boundary_miss_rate),
        "boundary_false_alarm_rate": float(boundary_false_alarm_rate),
        "boundary_precision": float(boundary_precision),
        "boundary_recall": float(boundary_recall),
        "boundary_f1": float(boundary_f1),
        "boundary_iou": float(boundary_iou),
    }


# ---------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------
def save_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_values(rows: List[Dict[str, object]], metric: str, freq: int, model_tag: str,
               threshold: float, tol: float) -> np.ndarray:
    vals = [
        float(r[metric]) for r in rows
        if int(r["freq_mhz"]) == int(freq)
        and r["model_tag"] == model_tag
        and abs(float(r["threshold_dbm"]) - threshold) <= tol
    ]
    vals = np.asarray(vals, dtype=float)
    return vals[np.isfinite(vals)]


def summarize_values(values: np.ndarray, cutoffs: List[float], prefix: str) -> Dict[str, object]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    out: Dict[str, object] = {
        f"{prefix}_n": int(len(values)),
        f"{prefix}_mean": np.nan,
        f"{prefix}_median": np.nan,
        f"{prefix}_p90": np.nan,
    }
    for c in cutoffs:
        out[f"{prefix}_F_le_{c:g}"] = np.nan

    if len(values) == 0:
        return out

    out[f"{prefix}_mean"] = float(np.mean(values))
    out[f"{prefix}_median"] = float(np.median(values))
    out[f"{prefix}_p90"] = float(np.percentile(values, 90))
    for c in cutoffs:
        out[f"{prefix}_F_le_{c:g}"] = float(np.mean(values <= c))
    return out


def build_summary(rows: List[Dict[str, object]],
                  freqs: List[int],
                  model_tags: List[str],
                  thresholds_by_freq: Dict[int, List[float]],
                  tol: float,
                  error_metric: str,
                  error_cutoffs: List[float],
                  miss_cutoffs: List[float]) -> List[Dict[str, object]]:
    summary = []
    for freq in freqs:
        for threshold in thresholds_by_freq[freq]:
            for tag in model_tags:
                err_vals = get_values(rows, error_metric, freq, tag, threshold, tol)
                miss_vals = get_values(rows, "boundary_miss_rate", freq, tag, threshold, tol)

                sub = [
                    r for r in rows
                    if int(r["freq_mhz"]) == int(freq)
                    and r["model_tag"] == tag
                    and abs(float(r["threshold_dbm"]) - threshold) <= tol
                ]

                boundary_counts = np.asarray([float(r["boundary_pixels"]) for r in sub], dtype=float)
                shadow_counts = np.asarray([float(r["boundary_gt_shadow_pixels"]) for r in sub], dtype=float)

                item: Dict[str, object] = {
                    "freq_mhz": freq,
                    "threshold_dbm": threshold,
                    "model_tag": tag,
                    "model_label": MODELS_CONFIG[tag]["label"],
                    "boundary_width_dbm": np.nan if not sub else sub[0]["boundary_width_dbm"],
                    "mean_boundary_pixels": float(np.mean(boundary_counts)) if len(boundary_counts) else np.nan,
                    "mean_boundary_gt_shadow_pixels": float(np.mean(shadow_counts)) if len(shadow_counts) else np.nan,
                }
                item.update(summarize_values(err_vals, error_cutoffs, error_metric))
                item.update(summarize_values(miss_vals, miss_cutoffs, "boundary_miss_rate"))
                summary.append(item)
    return summary


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_cdf_grid(rows: List[Dict[str, object]],
                  metric: str,
                  metric_title: str,
                  xlabel: str,
                  freqs: List[int],
                  model_tags: List[str],
                  thresholds_by_freq: Dict[int, List[float]],
                  threshold_label: str,
                  tol: float,
                  x_max: float,
                  x_step: float,
                  out_path: Path,
                  dpi: int) -> None:
    x_grid = np.arange(0.0, x_max + x_step, x_step)
    fig, axes = plt.subplots(1, len(freqs), figsize=(4.8 * len(freqs), 4.8), sharey=True)
    if len(freqs) == 1:
        axes = [axes]

    for ax, freq in zip(axes, freqs):
        # In fixed mode, there is usually one threshold per figure.
        # In per_freq mode, each frequency has its own single threshold.
        threshold = thresholds_by_freq[freq][0]

        for tag in model_tags:
            values = get_values(rows, metric, freq, tag, threshold, tol)
            if len(values) == 0:
                continue

            cdf = empirical_cdf(values, x_grid)
            ax.plot(
                x_grid,
                cdf,
                label=MODELS_CONFIG[tag]["label"],
                **MODEL_STYLE[tag],
            )

        ax.set_title(f"{freq} MHz\nT={threshold:.1f} dBm", fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("CDF", fontsize=11)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(model_tags), frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{metric_title} CDF | {threshold_label}", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_combined_grid(rows: List[Dict[str, object]],
                       error_metric: str,
                       freqs: List[int],
                       model_tags: List[str],
                       thresholds_by_freq: Dict[int, List[float]],
                       threshold_label: str,
                       tol: float,
                       error_x_max: float,
                       error_x_step: float,
                       miss_x_max: float,
                       miss_x_step: float,
                       out_path: Path,
                       dpi: int) -> None:
    x_error = np.arange(0.0, error_x_max + error_x_step, error_x_step)
    x_miss = np.arange(0.0, miss_x_max + miss_x_step, miss_x_step)

    fig, axes = plt.subplots(2, len(freqs), figsize=(4.8 * len(freqs), 8.8), sharey="row")
    if len(freqs) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, freq in enumerate(freqs):
        threshold = thresholds_by_freq[freq][0]

        for tag in model_tags:
            err_vals = get_values(rows, error_metric, freq, tag, threshold, tol)
            miss_vals = get_values(rows, "boundary_miss_rate", freq, tag, threshold, tol)

            if len(err_vals) > 0:
                axes[0, col].plot(
                    x_error,
                    empirical_cdf(err_vals, x_error),
                    label=MODELS_CONFIG[tag]["label"],
                    **MODEL_STYLE[tag],
                )

            if len(miss_vals) > 0:
                axes[1, col].plot(
                    x_miss,
                    empirical_cdf(miss_vals, x_miss),
                    label=MODELS_CONFIG[tag]["label"],
                    **MODEL_STYLE[tag],
                )

        axes[0, col].set_title(f"{freq} MHz\nT={threshold:.1f} dBm", fontsize=12, fontweight="bold")
        axes[0, col].set_xlabel("Boundary RSS Error (dB)", fontsize=10)
        axes[0, col].set_xlim(0, error_x_max)
        axes[0, col].set_ylim(0, 1.0)
        axes[0, col].grid(True, alpha=0.3)

        axes[1, col].set_xlabel("Boundary Miss Rate", fontsize=10)
        axes[1, col].set_xlim(0, miss_x_max)
        axes[1, col].set_ylim(0, 1.0)
        axes[1, col].grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("CDF\n(Error)", fontsize=11)
    axes[1, 0].set_ylabel("CDF\n(Miss Rate)", fontsize=11)

    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(model_tags), frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Boundary-only Error and Miss Rate CDF | {threshold_label}",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------
def evaluate_boundary_metrics(cfg: argparse.Namespace) -> List[Dict[str, object]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    rows: List[Dict[str, object]] = []

    for freq in cfg.freqs:
        print("\n" + "=" * 78)
        print(f"Frequency: {freq} MHz")
        print("=" * 78)

        thresholds = get_thresholds_for_freq(freq, cfg)
        models = load_models_for_freq(cfg.ckpt_dir, freq, cfg.models, device)
        if not models:
            print(f"No checkpoint found for {freq} MHz. Skip.")
            continue

        _, _, te_loader = get_dataloaders(
            root=cfg.root,
            freq=freq,
            img_size=cfg.img_size,
            batch_size=1,
            num_workers=cfg.num_workers,
        )

        with torch.no_grad():
            for sample_idx, (x, y, sample_n) in enumerate(te_loader):
                x = x.to(device)
                y = y.to(device)

                sample_id = int(sample_n[0].item()) if hasattr(sample_n[0], "item") else int(sample_n[0])
                building = x[0, 0].detach().cpu().numpy()
                gt_dbm = denorm(y[0, 0].detach().cpu().numpy(), cfg.min_dbm, cfg.max_dbm)

                for tag, model in models.items():
                    _, fine, _ = model(x)
                    pred_dbm = denorm(fine[0, 0].detach().cpu().numpy(), cfg.min_dbm, cfg.max_dbm)

                    for threshold in thresholds:
                        bm = compute_boundary_metrics(
                            pred_dbm=pred_dbm,
                            gt_dbm=gt_dbm,
                            building=building,
                            threshold=threshold,
                            boundary_width_dbm=cfg.boundary_width_dbm,
                        )

                        row = {
                            "freq_mhz": freq,
                            "sample_id": sample_id,
                            "sample_rank_in_test_loader": sample_idx,
                            "model_tag": tag,
                            "model_label": MODELS_CONFIG[tag]["label"],
                            "threshold_mode": cfg.threshold_mode,
                            "threshold_dbm": float(threshold),
                            "boundary_width_dbm": float(cfg.boundary_width_dbm),
                        }
                        row.update(bm)
                        rows.append(row)

                if (sample_idx + 1) % 50 == 0:
                    print(f"  {sample_idx + 1}/{len(te_loader)} test samples done")

    return rows


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = evaluate_boundary_metrics(cfg)

    if not rows:
        raise SystemExit("No rows generated. Check root, ckpt_dir, and selected models/frequencies.")

    # Save per-sample boundary metrics
    per_sample_fields = [
        "freq_mhz", "sample_id", "sample_rank_in_test_loader",
        "model_tag", "model_label", "threshold_mode", "threshold_dbm", "boundary_width_dbm",
        "boundary_pixels", "boundary_gt_shadow_pixels", "boundary_gt_nonshadow_pixels",
        "boundary_TP", "boundary_FP", "boundary_FN", "boundary_TN",
        "boundary_mae_db", "boundary_rmse_db", "boundary_bias_db", "boundary_abs_p90_db",
        "boundary_miss_rate", "boundary_false_alarm_rate",
        "boundary_precision", "boundary_recall", "boundary_f1", "boundary_iou",
    ]
    per_sample_path = out_dir / "per_sample_boundary_metrics.csv"
    save_csv(per_sample_path, rows, per_sample_fields)
    print(f"Saved: {per_sample_path}")

    # Prepare thresholds for plotting
    if cfg.threshold_mode == "fixed":
        threshold_plot_items = [(thr, {freq: [float(thr)] for freq in cfg.freqs}) for thr in cfg.thresholds]
    else:
        threshold_plot_items = [(None, {freq: [float(FREQ_THRESHOLD[freq])] for freq in cfg.freqs})]

    # Summary table
    if cfg.threshold_mode == "fixed":
        summary_thresholds_by_freq = {freq: [float(t) for t in cfg.thresholds] for freq in cfg.freqs}
    else:
        summary_thresholds_by_freq = {freq: [float(FREQ_THRESHOLD[freq])] for freq in cfg.freqs}

    summary_rows = build_summary(
        rows=rows,
        freqs=cfg.freqs,
        model_tags=cfg.models,
        thresholds_by_freq=summary_thresholds_by_freq,
        tol=cfg.threshold_tol,
        error_metric=cfg.error_metric,
        error_cutoffs=cfg.summary_error_cutoffs,
        miss_cutoffs=cfg.summary_miss_cutoffs,
    )

    summary_fields = [
        "freq_mhz", "threshold_dbm", "model_tag", "model_label", "boundary_width_dbm",
        "mean_boundary_pixels", "mean_boundary_gt_shadow_pixels",
        f"{cfg.error_metric}_n", f"{cfg.error_metric}_mean", f"{cfg.error_metric}_median", f"{cfg.error_metric}_p90",
        "boundary_miss_rate_n", "boundary_miss_rate_mean", "boundary_miss_rate_median", "boundary_miss_rate_p90",
    ]
    summary_fields += [f"{cfg.error_metric}_F_le_{c:g}" for c in cfg.summary_error_cutoffs]
    summary_fields += [f"boundary_miss_rate_F_le_{c:g}" for c in cfg.summary_miss_cutoffs]

    summary_path = out_dir / "boundary_cdf_summary.csv"
    save_csv(summary_path, summary_rows, summary_fields)
    print(f"Saved: {summary_path}")

    bw = width_tag(cfg.boundary_width_dbm)

    # Plot figures
    for thr, thresholds_by_freq in threshold_plot_items:
        ttag = threshold_tag(thr, cfg.threshold_mode)
        if cfg.threshold_mode == "fixed":
            threshold_label = f"Fixed threshold = {float(thr):.1f} dBm, boundary = ±{cfg.boundary_width_dbm:g} dB"
        else:
            threshold_label = f"Per-frequency threshold, boundary = ±{cfg.boundary_width_dbm:g} dB"

        error_path = out_dir / f"fig_boundary_error_cdf_th{ttag}_bw{bw}.png"
        miss_path = out_dir / f"fig_boundary_missrate_cdf_th{ttag}_bw{bw}.png"
        combined_path = out_dir / f"fig_boundary_combined_cdf_th{ttag}_bw{bw}.png"

        plot_cdf_grid(
            rows=rows,
            metric=cfg.error_metric,
            metric_title="Boundary-only RSS Error",
            xlabel="Sample-wise Boundary RSS Error (dB)",
            freqs=cfg.freqs,
            model_tags=cfg.models,
            thresholds_by_freq=thresholds_by_freq,
            threshold_label=threshold_label,
            tol=cfg.threshold_tol,
            x_max=cfg.error_x_max,
            x_step=cfg.error_x_step,
            out_path=error_path,
            dpi=cfg.dpi,
        )

        plot_cdf_grid(
            rows=rows,
            metric="boundary_miss_rate",
            metric_title="Boundary-only Miss Rate",
            xlabel="Sample-wise Boundary Miss Rate",
            freqs=cfg.freqs,
            model_tags=cfg.models,
            thresholds_by_freq=thresholds_by_freq,
            threshold_label=threshold_label,
            tol=cfg.threshold_tol,
            x_max=cfg.miss_x_max,
            x_step=cfg.miss_x_step,
            out_path=miss_path,
            dpi=cfg.dpi,
        )

        plot_combined_grid(
            rows=rows,
            error_metric=cfg.error_metric,
            freqs=cfg.freqs,
            model_tags=cfg.models,
            thresholds_by_freq=thresholds_by_freq,
            threshold_label=threshold_label,
            tol=cfg.threshold_tol,
            error_x_max=cfg.error_x_max,
            error_x_step=cfg.error_x_step,
            miss_x_max=cfg.miss_x_max,
            miss_x_step=cfg.miss_x_step,
            out_path=combined_path,
            dpi=cfg.dpi,
        )

    readme = out_dir / "README_boundary_cdf.txt"
    readme.write_text(
        "\n".join([
            "[boundary_only_cdf_analysis]",
            f"- threshold mode: {cfg.threshold_mode}",
            f"- thresholds: {cfg.thresholds if cfg.threshold_mode == 'fixed' else 'per-frequency thresholds'}",
            f"- boundary width: ±{cfg.boundary_width_dbm:g} dB around threshold",
            f"- error metric for CDF: {cfg.error_metric}",
            "- boundary pixels: valid pixels satisfying |GT_RSS - threshold| <= boundary_width_dbm",
            "- boundary-only error CDF: CDF of sample-wise boundary MAE/RMSE",
            "- boundary-only Miss Rate CDF: CDF of FN / GT-shadow pixels within boundary pixels",
        ]),
        encoding="utf-8",
    )
    print(f"Saved: {readme}")
    print("All done.")


if __name__ == "__main__":
    main()
