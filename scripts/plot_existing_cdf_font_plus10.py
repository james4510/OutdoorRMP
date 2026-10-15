#!/usr/bin/env python3
"""
Replot existing CDF result CSVs with larger fonts.

This script does not run model inference. It only reads existing metric CSVs
and regenerates RSS Error / Miss Rate CDF figures with font sizes increased
for paper or presentation inspection.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = [
    "radiounet-small",
    "radiounet-medium",
    "transpathnet-small",
    "transpathnet-base",
]

MODEL_LABELS = {
    "radiounet-small": "RadioUNet-S (35M)",
    "radiounet-medium": "RadioUNet-M (97M)",
    "transpathnet-small": "TransPathNet-S (38M)",
    "transpathnet-base": "TransPathNet-B (111M)",
}

MODEL_STYLE = {
    "radiounet-small": {"color": "tab:blue", "linestyle": "-", "linewidth": 2.8},
    "radiounet-medium": {"color": "tab:orange", "linestyle": "-", "linewidth": 2.8},
    "transpathnet-small": {"color": "tab:green", "linestyle": "--", "linewidth": 2.8},
    "transpathnet-base": {"color": "tab:red", "linestyle": "--", "linewidth": 2.8},
}

FS = {
    "axis_title": 21,
    "axis_label": 20,
    "boundary_axis_title": 22,
    "boundary_axis_label": 21,
    "tick": 18,
    "legend": 18,
    "boundary_legend": 20,
    "suptitle": 25,
    "boundary_suptitle": 26,
    "zoom_title": 23,
}

REPRESENTATIVE_DIR = Path(__file__).resolve().parents[1] / "results" / "representative_1750_4750_5750"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--metric_csv",
        default=str(REPRESENTATIVE_DIR / "metrics/per_sample_metrics_fixed_threshold.csv"),
    )
    p.add_argument(
        "--boundary_csv",
        default=str(REPRESENTATIVE_DIR / "boundary_cdf/per_sample_boundary_metrics.csv"),
    )
    p.add_argument(
        "--miss_out_dir",
        default=str(REPRESENTATIVE_DIR / "miss_cdf_font_plus10"),
    )
    p.add_argument(
        "--boundary_out_dir",
        default=str(REPRESENTATIVE_DIR / "boundary_cdf_font_plus10"),
    )
    p.add_argument("--freqs", type=int, nargs="+", default=[1750, 4750, 5750])
    p.add_argument("--zoom_freqs", type=int, nargs="+", default=[1750])
    p.add_argument("--thresholds", type=float, nargs="+", default=[-70.0, -65.0, -60.0])
    p.add_argument("--threshold_tol", type=float, default=1e-6)
    p.add_argument("--miss_x_max", type=float, default=1.0)
    p.add_argument("--miss_x_step", type=float, default=0.0025)
    p.add_argument("--error_x_max", type=float, default=10.0)
    p.add_argument("--error_x_step", type=float, default=0.05)
    p.add_argument("--boundary_width_dbm", type=float, default=10.0)
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def threshold_tag(threshold: float) -> str:
    if float(threshold).is_integer():
        return str(int(threshold))
    return str(threshold).replace(".", "p").replace("-", "m")


def width_tag(width: float) -> str:
    if float(width).is_integer():
        return str(int(width))
    return str(width).replace(".", "p")


def match_threshold(value: object, threshold: float, tol: float) -> bool:
    return abs(float(value) - float(threshold)) <= tol


def empirical_cdf(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.zeros_like(x_grid)
    values = np.sort(values)
    return np.searchsorted(values, x_grid, side="right") / len(values)


def sample_miss_values(rows: List[Dict[str, str]], freq: int, tag: str, threshold: float, tol: float) -> np.ndarray:
    values = []
    for row in rows:
        if int(float(row["freq_mhz"])) != int(freq):
            continue
        if row["model_tag"] != tag:
            continue
        if not match_threshold(row["threshold_dbm"], threshold, tol):
            continue

        if row.get("miss_rate", "") != "":
            values.append(float(row["miss_rate"]))
        else:
            tp = float(row["TP"])
            fn = float(row["FN"])
            values.append(np.nan if tp + fn == 0 else fn / (tp + fn))

    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def boundary_values(
    rows: List[Dict[str, str]],
    metric: str,
    freq: int,
    tag: str,
    threshold: float,
    tol: float,
) -> np.ndarray:
    values = [
        float(row[metric])
        for row in rows
        if int(float(row["freq_mhz"])) == int(freq)
        and row["model_tag"] == tag
        and match_threshold(row["threshold_dbm"], threshold, tol)
    ]
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def set_common_axis_style(ax, tick_size: int) -> None:
    ax.grid(True, alpha=0.32)
    ax.tick_params(axis="both", labelsize=tick_size)


def plot_sample_miss_main(
    rows: List[Dict[str, str]],
    freqs: List[int],
    threshold: float,
    tol: float,
    out_dir: Path,
    x_max: float,
    x_step: float,
    dpi: int,
) -> Path:
    x_grid = np.arange(0.0, x_max + x_step, x_step)
    fig, axes = plt.subplots(1, len(freqs), figsize=(7.0 * len(freqs), 7.6), sharex=True, sharey=True)
    if len(freqs) == 1:
        axes = [axes]

    for ax, freq in zip(axes, freqs):
        means = {}
        for tag in MODELS:
            vals = sample_miss_values(rows, freq, tag, threshold, tol)
            if len(vals) == 0:
                continue
            means[tag] = float(np.mean(vals))
            ax.plot(x_grid, empirical_cdf(vals, x_grid), label=MODEL_LABELS[tag], **MODEL_STYLE[tag])

        gap_txt = "N/A"
        if len(means) >= 2:
            gap_txt = f"Delta mean={max(means.values()) - min(means.values()):.3f}"

        ax.set_title(f"{freq} MHz\n{gap_txt}", fontsize=FS["axis_title"], fontweight="bold")
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.0)
        ax.set_box_aspect(0.72)
        ax.set_xticks(np.arange(0.0, x_max + 1e-9, 0.10))
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        ax.tick_params(axis="x", labelrotation=45)
        set_common_axis_style(ax, FS["tick"])
        ax.set_xlabel("Sample-wise Miss Rate", fontsize=FS["axis_label"])

    axes[0].set_ylabel("CDF", fontsize=FS["axis_label"])
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(MODELS),
        fontsize=FS["legend"],
        frameon=True,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.suptitle(
        f"Sample-wise Miss Rate CDF | Fixed Threshold {threshold:.1f} dBm",
        fontsize=FS["suptitle"],
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.25, 1, 0.88])

    out_path = out_dir / f"fig_cdf_missrate_main_th{threshold_tag(threshold)}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def choose_zoom_window(
    rows: List[Dict[str, str]],
    freq: int,
    threshold: float,
    tol: float,
    x_max: float,
    x_step: float,
) -> Tuple[float, float, float, float]:
    x_grid = np.arange(0.0, x_max + x_step, x_step)
    curves = []
    for tag in MODELS:
        vals = sample_miss_values(rows, freq, tag, threshold, tol)
        if len(vals) > 0:
            curves.append(empirical_cdf(vals, x_grid))

    if len(curves) < 2:
        return 0.0, min(0.25, x_max), 0.0, 1.0

    stack = np.vstack(curves)
    spread = stack.max(axis=0) - stack.min(axis=0)
    x_star = float(x_grid[int(np.argmax(spread))])
    x0 = max(0.0, x_star - 0.07)
    x1 = min(x_max, x_star + 0.07)
    if x1 - x0 < 0.08:
        x0 = max(0.0, x0 - 0.04)
        x1 = min(x_max, x1 + 0.04)

    in_window = (x_grid >= x0) & (x_grid <= x1)
    y0 = max(0.0, float(np.min(stack[:, in_window])) - 0.06)
    y1 = min(1.0, float(np.max(stack[:, in_window])) + 0.06)
    if y1 - y0 < 0.18:
        mid = 0.5 * (y0 + y1)
        y0 = max(0.0, mid - 0.10)
        y1 = min(1.0, mid + 0.10)
    return x0, x1, y0, y1


def plot_sample_miss_zoom(
    rows: List[Dict[str, str]],
    freq: int,
    threshold: float,
    tol: float,
    out_dir: Path,
    x_max: float,
    x_step: float,
    dpi: int,
) -> Path:
    x_grid = np.arange(0.0, x_max + x_step, x_step)
    x0, x1, y0, y1 = choose_zoom_window(rows, freq, threshold, tol, x_max, x_step)

    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    for tag in MODELS:
        vals = sample_miss_values(rows, freq, tag, threshold, tol)
        if len(vals) == 0:
            continue
        ax.plot(x_grid, empirical_cdf(vals, x_grid), label=MODEL_LABELS[tag], **MODEL_STYLE[tag])

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks(np.linspace(x0, x1, 6))
    ax.set_yticks(np.linspace(y0, y1, 6))
    set_common_axis_style(ax, FS["tick"])
    ax.set_xlabel("Sample-wise Miss Rate", fontsize=FS["axis_label"])
    ax.set_ylabel("CDF", fontsize=FS["axis_label"])
    ax.legend(fontsize=FS["legend"], loc="lower right", framealpha=0.95)
    ax.set_title(
        f"Zoomed Miss Rate CDF | {freq} MHz | Threshold {threshold:.1f} dBm",
        fontsize=FS["zoom_title"],
        fontweight="bold",
    )
    fig.tight_layout()

    out_path = out_dir / f"fig_cdf_missrate_zoom_{freq}MHz_th{threshold_tag(threshold)}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_boundary_grid(
    rows: List[Dict[str, str]],
    metric: str,
    metric_title: str,
    xlabel: str,
    freqs: List[int],
    threshold: float,
    tol: float,
    x_max: float,
    x_step: float,
    out_path: Path,
    dpi: int,
) -> Path:
    x_grid = np.arange(0.0, x_max + x_step, x_step)
    fig, axes = plt.subplots(1, len(freqs), figsize=(7.0 * len(freqs), 6.5), sharey=True)
    if len(freqs) == 1:
        axes = [axes]

    for ax, freq in zip(axes, freqs):
        for tag in MODELS:
            vals = boundary_values(rows, metric, freq, tag, threshold, tol)
            if len(vals) == 0:
                continue
            ax.plot(x_grid, empirical_cdf(vals, x_grid), label=MODEL_LABELS[tag], **MODEL_STYLE[tag])

        ax.set_title(f"{freq} MHz\nT={threshold:.1f} dBm", fontsize=FS["boundary_axis_title"], fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=FS["boundary_axis_label"])
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        set_common_axis_style(ax, FS["tick"])

    axes[0].set_ylabel("CDF", fontsize=FS["boundary_axis_label"])
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(MODELS),
        fontsize=FS["boundary_legend"],
        frameon=True,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle(
        f"{metric_title} CDF | Fixed threshold = {threshold:.1f} dBm, boundary = +/-10 dB",
        fontsize=FS["boundary_suptitle"],
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.88])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_boundary_combined(
    rows: List[Dict[str, str]],
    freqs: List[int],
    threshold: float,
    tol: float,
    error_x_max: float,
    error_x_step: float,
    miss_x_max: float,
    miss_x_step: float,
    out_path: Path,
    dpi: int,
) -> Path:
    x_error = np.arange(0.0, error_x_max + error_x_step, error_x_step)
    x_miss = np.arange(0.0, miss_x_max + miss_x_step, miss_x_step)

    fig, axes = plt.subplots(2, len(freqs), figsize=(7.75 * len(freqs), 12.2), sharey="row")
    if len(freqs) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, freq in enumerate(freqs):
        for tag in MODELS:
            err_vals = boundary_values(rows, "boundary_mae_db", freq, tag, threshold, tol)
            miss_vals = boundary_values(rows, "boundary_miss_rate", freq, tag, threshold, tol)
            if len(err_vals) > 0:
                axes[0, col].plot(x_error, empirical_cdf(err_vals, x_error), label=MODEL_LABELS[tag], **MODEL_STYLE[tag])
            if len(miss_vals) > 0:
                axes[1, col].plot(x_miss, empirical_cdf(miss_vals, x_miss), label=MODEL_LABELS[tag], **MODEL_STYLE[tag])

        axes[0, col].set_title(f"{freq} MHz\nT={threshold:.1f} dBm", fontsize=FS["boundary_axis_title"], fontweight="bold")
        axes[0, col].set_xlabel("Threshold-margin RSS Error (dB)", fontsize=FS["boundary_axis_label"])
        axes[0, col].set_xlim(0, error_x_max)
        axes[0, col].set_ylim(0, 1.0)
        axes[0, col].set_yticks(np.linspace(0.0, 1.0, 6))
        set_common_axis_style(axes[0, col], FS["tick"])

        axes[1, col].set_xlabel("Threshold-margin Miss Rate", fontsize=FS["boundary_axis_label"])
        axes[1, col].set_xlim(0, miss_x_max)
        axes[1, col].set_ylim(0, 1.0)
        axes[1, col].set_yticks(np.linspace(0.0, 1.0, 6))
        set_common_axis_style(axes[1, col], FS["tick"])

    axes[0, 0].set_ylabel("CDF\n(Error)", fontsize=FS["boundary_axis_label"])
    axes[1, 0].set_ylabel("CDF\n(Miss Rate)", fontsize=FS["boundary_axis_label"])

    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(MODELS),
        fontsize=FS["boundary_legend"],
        frameon=True,
        bbox_to_anchor=(0.5, -0.025),
    )
    fig.suptitle(
        f"Boundary-only Error and Miss Rate CDF | Fixed threshold = {threshold:.1f} dBm, boundary = +/-10 dB",
        fontsize=FS["boundary_suptitle"],
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.91], w_pad=4.8, h_pad=2.2)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def validate_thresholds(rows: List[Dict[str, str]], thresholds: Iterable[float], tol: float, label: str) -> None:
    existing = sorted({float(row["threshold_dbm"]) for row in rows})
    print(f"{label} thresholds in CSV: {existing}")
    for threshold in thresholds:
        if not any(match_threshold(x, threshold, tol) for x in existing):
            print(f"[WARN] {label}: threshold {threshold:.1f} dBm is not present.")


def main() -> None:
    args = parse_args()
    metric_rows = read_csv(Path(args.metric_csv))
    boundary_rows = read_csv(Path(args.boundary_csv))

    miss_out = Path(args.miss_out_dir)
    boundary_out = Path(args.boundary_out_dir)
    miss_out.mkdir(parents=True, exist_ok=True)
    boundary_out.mkdir(parents=True, exist_ok=True)

    validate_thresholds(metric_rows, args.thresholds, args.threshold_tol, "metric")
    validate_thresholds(boundary_rows, args.thresholds, args.threshold_tol, "boundary")

    saved: List[Path] = []
    bw = width_tag(args.boundary_width_dbm)

    for threshold in args.thresholds:
        saved.append(
            plot_sample_miss_main(
                metric_rows,
                args.freqs,
                threshold,
                args.threshold_tol,
                miss_out,
                args.miss_x_max,
                args.miss_x_step,
                args.dpi,
            )
        )
        for freq in args.zoom_freqs:
            saved.append(
                plot_sample_miss_zoom(
                    metric_rows,
                    freq,
                    threshold,
                    args.threshold_tol,
                    miss_out,
                    args.miss_x_max,
                    args.miss_x_step,
                    args.dpi,
                )
            )

        tag = threshold_tag(threshold)
        saved.append(
            plot_boundary_grid(
                boundary_rows,
                "boundary_mae_db",
                "Boundary-only RSS Error",
                "Sample-wise Threshold-margin RSS Error (dB)",
                args.freqs,
                threshold,
                args.threshold_tol,
                args.error_x_max,
                args.error_x_step,
                boundary_out / f"fig_boundary_error_cdf_x10_th{tag}_bw{bw}.png",
                args.dpi,
            )
        )
        saved.append(
            plot_boundary_grid(
                boundary_rows,
                "boundary_miss_rate",
                "Boundary-only Miss Rate",
                "Sample-wise Boundary Miss Rate",
                args.freqs,
                threshold,
                args.threshold_tol,
                args.miss_x_max,
                args.miss_x_step,
                boundary_out / f"fig_boundary_missrate_cdf_th{tag}_bw{bw}.png",
                args.dpi,
            )
        )
        saved.append(
            plot_boundary_combined(
                boundary_rows,
                args.freqs,
                threshold,
                args.threshold_tol,
                args.error_x_max,
                args.error_x_step,
                args.miss_x_max,
                args.miss_x_step,
                boundary_out / f"fig_boundary_combined_cdf_x10_th{tag}_bw{bw}.png",
                args.dpi,
            )
        )

    print("Saved files:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
