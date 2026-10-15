#!/usr/bin/env python3
"""
Replot a fixed-threshold metric heatmap from an existing summary CSV.

This script does not run model inference. It only reads
summary_metrics_fixed_threshold.csv and redraws the metric heatmap with larger
fonts.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = [
    "radiounet-small",
    "radiounet-medium",
    "transpathnet-small",
    "transpathnet-base",
]

MODEL_LABELS = {
    "radiounet-small": "RadioUNet-S",
    "radiounet-medium": "RadioUNet-M",
    "transpathnet-small": "TransPathNet-S",
    "transpathnet-base": "TransPathNet-B",
}

METRICS = ["precision", "recall", "iou", "f1"]
METRIC_TITLES = {
    "precision": "Precision",
    "recall": "Recall",
    "iou": "IoU",
    "f1": "F1-score",
}

FS = {
    "suptitle": 74,
    "metric_title": 64,
    "axis_label": 58,
    "tick": 52,
    "cell": 50,
    "cbar_label": 54,
    "cbar_tick": 50,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--summary_csv",
        default="github_release_radio_map_shadow_detection/results/fixed_threshold_metrics/summary_metrics_fixed_threshold.csv",
    )
    p.add_argument("--threshold", type=float, default=-65.0)
    p.add_argument("--freqs", type=int, nargs="+", default=[1750, 4750, 5750])
    p.add_argument("--models", nargs="+", default=MODEL_ORDER, choices=MODEL_ORDER)
    p.add_argument("--out", default="github_release_radio_map_shadow_detection/results/fixed_threshold_th-65.png")
    p.add_argument(
        "--out_copy",
        default="github_release_radio_map_shadow_detection/results/fixed_threshold_metrics/heatmap_fixed_threshold_th-65_font_plus10.png",
    )
    p.add_argument("--vmin", type=float, default=0.0)
    p.add_argument("--vmax", type=float, default=1.0)
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def read_summary(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def threshold_tag(threshold: float) -> str:
    if float(threshold).is_integer():
        return str(int(threshold))
    return str(threshold).replace(".", "p").replace("-", "m")


def value_from_summary(
    summary: List[Dict[str, str]],
    threshold: float,
    freq: int,
    model_tag: str,
    metric: str,
) -> float:
    for row in summary:
        if (
            abs(float(row["threshold_dbm"]) - float(threshold)) < 1e-6
            and int(float(row["freq_mhz"])) == int(freq)
            and row["model_tag"] == model_tag
        ):
            return float(row[f"{metric}_mean"])
    return np.nan


def plot_heatmap(args: argparse.Namespace) -> List[Path]:
    summary = read_summary(Path(args.summary_csv))
    fig, axes = plt.subplots(2, 2, figsize=(36.0, 24.0))
    axes = axes.flatten()

    for plot_idx, (ax, metric) in enumerate(zip(axes, METRICS)):
        mat = np.zeros((len(args.freqs), len(args.models)), dtype=float)
        for i, freq in enumerate(args.freqs):
            for j, tag in enumerate(args.models):
                mat[i, j] = value_from_summary(summary, args.threshold, freq, tag, metric)

        im = ax.imshow(mat, vmin=args.vmin, vmax=args.vmax, cmap=args.cmap, aspect="auto")

        ax.set_title(METRIC_TITLES[metric], fontsize=FS["metric_title"], fontweight="bold", pad=24)
        ax.set_xticks(np.arange(len(args.models)))
        if plot_idx >= 2:
            ax.set_xticklabels(
                [MODEL_LABELS[tag] for tag in args.models],
                fontsize=FS["tick"],
                rotation=45,
                ha="right",
                rotation_mode="anchor",
            )
            ax.set_xlabel("Model", fontsize=FS["axis_label"], labelpad=62)
        else:
            ax.set_xticklabels([])
            ax.set_xlabel("")
        ax.set_yticks(np.arange(len(args.freqs)))
        ax.set_yticklabels(
            [f"{freq}" for freq in args.freqs],
            fontsize=FS["tick"],
            rotation=45,
            ha="right",
            va="center",
            rotation_mode="anchor",
        )
        ax.set_ylabel("Frequency (MHz)", fontsize=FS["axis_label"], labelpad=78)
        ax.tick_params(axis="both", labelsize=FS["tick"])

        for i in range(len(args.freqs)):
            row_vals = mat[i, :]
            best_j = int(np.nanargmax(row_vals)) if np.isfinite(row_vals).any() else -1
            for j in range(len(args.models)):
                val = mat[i, j]
                text = "N/A" if not np.isfinite(val) else f"{val:.3f}"
                weight = "bold" if j == best_j else "normal"
                color = "white" if np.isfinite(val) and val < (args.vmin + args.vmax) / 2 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=FS["cell"], fontweight=weight, color=color)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
        cbar.set_label("Score", fontsize=FS["cbar_label"], labelpad=22)
        cbar.ax.tick_params(labelsize=FS["cbar_tick"])

    fig.suptitle(
        f"Shadow Detection Metrics under Fixed Threshold {args.threshold:.1f} dBm",
        fontsize=FS["suptitle"],
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.92], w_pad=16.0, h_pad=7.0)

    out_paths = [Path(args.out)]
    if args.out_copy:
        out_paths.append(Path(args.out_copy))

    for path in out_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close(fig)
    return out_paths


def main() -> None:
    args = parse_args()
    if args.out == "github_release_radio_map_shadow_detection/results/fixed_threshold_th-65.png":
        args.out = f"github_release_radio_map_shadow_detection/results/fixed_threshold_th{threshold_tag(args.threshold)}.png"
    if args.out_copy == "github_release_radio_map_shadow_detection/results/fixed_threshold_metrics/heatmap_fixed_threshold_th-65_font_plus10.png":
        args.out_copy = (
            "github_release_radio_map_shadow_detection/results/fixed_threshold_metrics/"
            f"heatmap_fixed_threshold_th{threshold_tag(args.threshold)}_font_plus10.png"
        )
    plot_heatmap(args)


if __name__ == "__main__":
    main()
