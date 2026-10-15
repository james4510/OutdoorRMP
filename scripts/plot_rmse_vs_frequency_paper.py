#!/usr/bin/env python3
"""Replot the RMSE-vs-frequency figure with paper column-width fonts."""

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

MODEL_STYLE = {
    "radiounet-small": {"color": "tab:blue", "linestyle": "-", "marker": "o"},
    "radiounet-medium": {"color": "tab:orange", "linestyle": "-", "marker": "s"},
    "transpathnet-small": {"color": "tab:green", "linestyle": "--", "marker": "^"},
    "transpathnet-base": {"color": "tab:red", "linestyle": "--", "marker": "D"},
}

FS = {
    "title": 34,
    "axis_label": 30,
    "tick": 28,
    "legend": 24,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--summary_csv",
        default="ikeee/final_results/metrics_fixed_threshold/summary_metrics_long.csv",
    )
    p.add_argument("--threshold", type=float, default=-65.0)
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument(
        "--out",
        nargs="+",
        default=[
            "ikeee/final_results/metrics_fixed_threshold/fig_rmse_vs_frequency.png",
            "ikeee/final_results/metrics/fig_rmse_vs_frequency.png",
        ],
    )
    return p.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.summary_csv))
    rows = [row for row in rows if abs(float(row["threshold_dbm"]) - args.threshold) < 1e-6]

    freqs = sorted({int(float(row["freq_mhz"])) for row in rows})
    fig, ax = plt.subplots(figsize=(12.0, 7.0))

    for tag in MODEL_ORDER:
        model_rows = {
            int(float(row["freq_mhz"])): row
            for row in rows
            if row["model_tag"] == tag
        }
        if not model_rows:
            continue

        y = np.array([float(model_rows[freq]["rmse_dbm_mean"]) for freq in freqs], dtype=float)
        yerr = np.array([float(model_rows[freq]["rmse_dbm_std"]) for freq in freqs], dtype=float)
        label = model_rows[freqs[0]]["model_label"]
        ax.errorbar(
            freqs,
            y,
            yerr=yerr,
            label=label,
            linewidth=3.2,
            markersize=9,
            capsize=6,
            capthick=2.4,
            elinewidth=2.4,
            **MODEL_STYLE[tag],
        )

    ax.set_title("RSS Prediction Error across Frequencies", fontsize=FS["title"], fontweight="bold", pad=18)
    ax.set_xlabel("Frequency (MHz)", fontsize=FS["axis_label"], labelpad=12)
    ax.set_ylabel("RMSE (dBm)", fontsize=FS["axis_label"], labelpad=12)
    ax.set_xticks(freqs)
    ax.tick_params(axis="both", labelsize=FS["tick"])
    ax.grid(True, alpha=0.32)
    ax.set_ylim(1.05, 3.30)
    ax.legend(loc="lower right", fontsize=FS["legend"], framealpha=0.88)

    fig.tight_layout()

    for out in args.out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
