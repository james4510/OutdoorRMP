#!/usr/bin/env python3
"""
final_samplewise_cdf_v3.py

Revised sample-wise CDF analysis for paper figures.

Main design changes from v2
---------------------------
1) Remove the lower false-alarm row from the main figure.
   -> Keep only miss-rate CDF: FN / (TP + FN)
2) Use true CDF scaling on the y-axis (0 ~ 1).
3) Show the frequency-wise trend of model gaps clearly.
   -> Main figure is one row of frequencies (e.g., 1750 / 3750 / 5750 MHz)
      for each fixed threshold.
4) Add zoomed CDF figures for frequencies where model differences are subtle
   (e.g., 1750 MHz), using an automatically selected focus window around the
   largest model-gap region.
5) Support fixed thresholds such as -70 / -65 / -60 dBm.
   NOTE: the input CSV must already contain rows evaluated at those thresholds.

Input
-----
This script reads per_sample_metrics.csv generated elsewhere, and filters rows
by threshold_dbm.

Outputs
-------
- samplewise_error_rates_v3.csv
- cdf_summary_v3.csv
- fig_cdf_missrate_main_th{thr}.png          (main 1 x Nfreq figure)
- fig_cdf_missrate_zoom_{freq}MHz_th{thr}.png (zoomed figure)
- selected_zoom_windows_v3.csv

Example
-------
python final_samplewise_cdf_v3.py \
    --input_csv ./final_results/metrics/per_sample_metrics.csv \
    --result_dir ./final_results/cdf_v3 \
    --main_freqs 1750 3750 5750 \
    --zoom_freqs 1750 \
    --thresholds -70 -65 -60
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

FREQS = [1750, 2750, 3750, 4750, 5750]

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
    "radiounet-small":    {"ls": "-",  "color": "tab:blue"},
    "radiounet-medium":   {"ls": "-",  "color": "tab:orange"},
    "transpathnet-small": {"ls": "--", "color": "tab:green"},
    "transpathnet-base":  {"ls": "--", "color": "tab:red"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True, help="per_sample_metrics.csv from final_eval_metrics.py")
    p.add_argument("--result_dir", default="./final_results/cdf_v3")
    p.add_argument("--freqs", type=int, nargs="+", default=FREQS)
    p.add_argument("--main_freqs", type=int, nargs="+", default=[1750, 3750, 5750])
    p.add_argument("--zoom_freqs", type=int, nargs="+", default=[1750],
                   help="Frequencies for which an additional zoomed CDF is generated.")
    p.add_argument("--thresholds", type=float, nargs="+", default=[-70.0, -65.0, -60.0])
    p.add_argument("--x_max", type=float, default=1.0)
    p.add_argument("--x_step", type=float, default=0.0025)
    p.add_argument("--tick_step", type=float, default=0.10)
    p.add_argument("--zoom_half_width", type=float, default=0.07,
                   help="Half-width of the x zoom window around the largest-gap point.")
    p.add_argument("--zoom_margin_y", type=float, default=0.06,
                   help="Y-margin added around the zoomed CDF range.")
    p.add_argument("--threshold_tol", type=float, default=1e-6)
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_error_rows(records: List[Dict[str, str]]) -> List[Dict[str, object]]:
    rows = []
    for r in records:
        TP = int(float(r["TP"]))
        FP = int(float(r["FP"]))
        FN = int(float(r["FN"]))
        TN = int(float(r["TN"]))

        miss_denom = TP + FN
        false_alarm_denom = FP + TN

        miss_rate = np.nan if miss_denom == 0 else FN / miss_denom
        false_alarm_rate = np.nan if false_alarm_denom == 0 else FP / false_alarm_denom

        rows.append({
            "freq_mhz": int(float(r["freq_mhz"])),
            "sample_id": int(float(r["sample_id"])),
            "model_tag": r["model_tag"],
            "model_label": r.get("model_label", MODEL_LABELS.get(r["model_tag"], r["model_tag"])),
            "threshold_dbm": float(r["threshold_dbm"]),
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "miss_rate": float(miss_rate),
            "false_alarm_rate": float(false_alarm_rate),
            "recall": float(r["recall"]),
            "precision": float(r["precision"]),
            "f1": float(r["f1"]),
            "iou": float(r["iou"]),
            "rmse_dbm": float(r["rmse_dbm"]),
        })
    return rows


def match_threshold(val: float, threshold: float, tol: float) -> bool:
    return abs(float(val) - float(threshold)) <= tol


def get_values(rows: List[Dict[str, object]], freq: int, tag: str, metric: str,
               threshold: float, tol: float) -> np.ndarray:
    vals = [
        float(r[metric])
        for r in rows
        if int(r["freq_mhz"]) == freq
        and r["model_tag"] == tag
        and match_threshold(float(r["threshold_dbm"]), threshold, tol)
    ]
    arr = np.asarray(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    return arr


def cdf_fraction(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.zeros_like(x_grid)
    return np.array([np.mean(values <= x) for x in x_grid], dtype=float)


def sanity_check_threshold_coverage(rows: List[Dict[str, object]], thresholds: List[float], tol: float):
    existing = sorted({float(r["threshold_dbm"]) for r in rows})
    print(f"Detected threshold values in CSV: {existing}")
    for thr in thresholds:
        cnt = sum(1 for r in rows if match_threshold(float(r["threshold_dbm"]), thr, tol))
        if cnt == 0:
            print(f"[WARN] No rows found for threshold {thr:.1f} dBm.")


def compute_gap_summary(rows: List[Dict[str, object]], freqs: List[int], thresholds: List[float], tol: float) -> List[Dict[str, object]]:
    summary = []
    for thr in thresholds:
        for freq in freqs:
            per_model_means = {}
            for tag in MODELS:
                vals = get_values(rows, freq, tag, "miss_rate", thr, tol)
                per_model_means[tag] = np.nan if len(vals) == 0 else float(np.mean(vals))

            valid_means = [v for v in per_model_means.values() if np.isfinite(v)]
            gap = np.nan if not valid_means else float(np.nanmax(valid_means) - np.nanmin(valid_means))

            item = {
                "threshold_dbm": thr,
                "freq_mhz": freq,
                "gap_mean_miss_rate": gap,
            }
            for tag in MODELS:
                item[f"{tag}_mean_miss_rate"] = per_model_means[tag]
            summary.append(item)
    return summary


def plot_main_miss_cdf(rows: List[Dict[str, object]], freqs: List[int], threshold: float, tol: float,
                       out_dir: Path, x_max: float, x_step: float, tick_step: float, dpi: int):
    """
    Main figure: 1 x Nfreq layout, only miss-rate CDF.
    This makes the frequency-wise trend of model-gap progression easier to see.
    """
    x_grid = np.arange(0, x_max + x_step, x_step)
    xticks = np.arange(0, x_max + 1e-9, tick_step)

    fig, axes = plt.subplots(1, len(freqs), figsize=(4.6 * len(freqs), 4.8), sharex=True, sharey=True)
    if len(freqs) == 1:
        axes = [axes]

    for ax, freq in zip(axes, freqs):
        model_means = {}
        for tag in MODELS:
            vals = get_values(rows, freq, tag, "miss_rate", threshold, tol)
            if len(vals) == 0:
                continue
            y = cdf_fraction(vals, x_grid)
            st = MODEL_STYLE[tag]
            ax.plot(
                x_grid,
                y,
                color=st["color"],
                linestyle=st["ls"],
                linewidth=2.3,
                label=MODEL_LABELS[tag],
            )
            model_means[tag] = float(np.mean(vals))

        gap_txt = "N/A"
        if len(model_means) >= 2:
            gap_txt = f"Δmean={max(model_means.values()) - min(model_means.values()):.3f}"

        ax.set_title(f"{freq} MHz\n{gap_txt}", fontsize=11, fontweight="bold")
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(xticks)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Sample-wise miss rate", fontsize=10)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.95)

    axes[0].set_ylabel("CDF", fontsize=10)
    fig.suptitle(
        f"Sample-wise Miss Rate CDF | Fixed Threshold {threshold:.1f} dBm",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    thr_tag = f"{int(threshold)}" if float(threshold).is_integer() else str(threshold).replace('.', 'p')
    path = out_dir / f"fig_cdf_missrate_main_th{thr_tag}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def choose_zoom_window(rows: List[Dict[str, object]], freq: int, threshold: float, tol: float,
                       x_max: float, x_step: float, half_width: float, margin_y: float) -> Tuple[float, float, float, float]:
    """
    Automatically focus on the x-range where the model spread in CDF is maximal.
    This is useful for cases like 1750 MHz where model differences are small.
    """
    x_grid = np.arange(0, x_max + x_step, x_step)
    curves = []
    for tag in MODELS:
        vals = get_values(rows, freq, tag, "miss_rate", threshold, tol)
        if len(vals) == 0:
            continue
        curves.append(cdf_fraction(vals, x_grid))

    if len(curves) < 2:
        return 0.0, min(0.25, x_max), 0.0, 1.0

    stack = np.vstack(curves)
    spread = stack.max(axis=0) - stack.min(axis=0)
    best_idx = int(np.argmax(spread))
    x_star = float(x_grid[best_idx])

    x0 = max(0.0, x_star - half_width)
    x1 = min(x_max, x_star + half_width)
    if x1 - x0 < 0.08:
        x0 = max(0.0, x0 - 0.04)
        x1 = min(x_max, x1 + 0.04)

    in_window = (x_grid >= x0) & (x_grid <= x1)
    y_min = float(np.min(stack[:, in_window]))
    y_max = float(np.max(stack[:, in_window]))

    y0 = max(0.0, y_min - margin_y)
    y1 = min(1.0, y_max + margin_y)
    if y1 - y0 < 0.18:
        mid = 0.5 * (y0 + y1)
        y0 = max(0.0, mid - 0.10)
        y1 = min(1.0, mid + 0.10)

    return x0, x1, y0, y1


def plot_zoom_miss_cdf(rows: List[Dict[str, object]], freq: int, threshold: float, tol: float,
                       out_dir: Path, x_max: float, x_step: float, tick_step: float,
                       half_width: float, margin_y: float, dpi: int) -> Dict[str, object]:
    x_grid = np.arange(0, x_max + x_step, x_step)
    x0, x1, y0, y1 = choose_zoom_window(rows, freq, threshold, tol, x_max, x_step, half_width, margin_y)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    for tag in MODELS:
        vals = get_values(rows, freq, tag, "miss_rate", threshold, tol)
        if len(vals) == 0:
            continue
        y = cdf_fraction(vals, x_grid)
        st = MODEL_STYLE[tag]
        ax.plot(
            x_grid,
            y,
            color=st["color"],
            linestyle=st["ls"],
            linewidth=2.5,
            label=MODEL_LABELS[tag],
        )

    xticks = np.linspace(x0, x1, 6)
    yticks = np.linspace(y0, y1, 6)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.grid(True, alpha=0.35)
    ax.set_xlabel("Sample-wise miss rate", fontsize=10)
    ax.set_ylabel("CDF", fontsize=10)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    ax.set_title(
        f"Zoomed Miss Rate CDF | {freq} MHz | Threshold {threshold:.1f} dBm",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    thr_tag = f"{int(threshold)}" if float(threshold).is_integer() else str(threshold).replace('.', 'p')
    path = out_dir / f"fig_cdf_missrate_zoom_{freq}MHz_th{thr_tag}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    return {
        "threshold_dbm": threshold,
        "freq_mhz": freq,
        "x_min": x0,
        "x_max": x1,
        "y_min": y0,
        "y_max": y1,
    }


def main():
    cfg = parse_args()
    out_dir = Path(cfg.result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = read_csv(Path(cfg.input_csv))
    rows = build_error_rows(records)

    sanity_check_threshold_coverage(rows, cfg.thresholds, cfg.threshold_tol)

    # Save full processed rows.
    fields = [
        "freq_mhz", "sample_id", "model_tag", "model_label", "threshold_dbm",
        "TP", "FP", "FN", "TN",
        "miss_rate", "false_alarm_rate",
        "recall", "precision", "f1", "iou", "rmse_dbm",
    ]
    save_csv(out_dir / "samplewise_error_rates_v3.csv", rows, fields)
    print(f"Saved: {out_dir / 'samplewise_error_rates_v3.csv'}")

    # Gap summary table.
    summary = compute_gap_summary(rows, cfg.freqs, cfg.thresholds, cfg.threshold_tol)
    if summary:
        save_csv(out_dir / "cdf_summary_v3.csv", summary, list(summary[0].keys()))
        print(f"Saved: {out_dir / 'cdf_summary_v3.csv'}")

    # Main CDF figures: one figure per threshold.
    for thr in cfg.thresholds:
        plot_main_miss_cdf(
            rows=rows,
            freqs=cfg.main_freqs,
            threshold=thr,
            tol=cfg.threshold_tol,
            out_dir=out_dir,
            x_max=cfg.x_max,
            x_step=cfg.x_step,
            tick_step=cfg.tick_step,
            dpi=cfg.dpi,
        )

    # Zoom figures: typically for 1750 MHz where the curves are similar.
    zoom_rows = []
    for thr in cfg.thresholds:
        for freq in cfg.zoom_freqs:
            zoom_rows.append(
                plot_zoom_miss_cdf(
                    rows=rows,
                    freq=freq,
                    threshold=thr,
                    tol=cfg.threshold_tol,
                    out_dir=out_dir,
                    x_max=cfg.x_max,
                    x_step=cfg.x_step,
                    tick_step=cfg.tick_step,
                    half_width=cfg.zoom_half_width,
                    margin_y=cfg.zoom_margin_y,
                    dpi=cfg.dpi,
                )
            )

    if zoom_rows:
        save_csv(out_dir / "selected_zoom_windows_v3.csv", zoom_rows, list(zoom_rows[0].keys()))
        print(f"Saved: {out_dir / 'selected_zoom_windows_v3.csv'}")

    readme = out_dir / "README_v3.txt"
    readme.write_text(
        "\n".join([
            "[final_samplewise_cdf_v3]",
            "- Main figure keeps only miss-rate CDF: FN / (TP + FN)",
            "- Y-axis is normalized to 0~1 (true CDF)",
            f"- Thresholds: {cfg.thresholds}",
            f"- Main frequencies: {cfg.main_freqs}",
            f"- Zoom frequencies: {cfg.zoom_freqs}",
            "- IMPORTANT: if the requested thresholds are not present in the input CSV,",
            "  you must first regenerate per_sample_metrics.csv with those fixed thresholds.",
        ]),
        encoding="utf-8",
    )
    print(f"Saved: {readme}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
