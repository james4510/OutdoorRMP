#!/usr/bin/env python3
"""
Aggregate fixed-threshold metric summaries across repeated random seeds.

Expected layout from run_multi_seed_holdout_experiments.sh:

  <result_root>/
    seed_0/metrics/summary_metrics_fixed_threshold.csv
    seed_1/metrics/summary_metrics_fixed_threshold.csv
    ...

Outputs:
  - summary_metrics_by_seed.csv
  - summary_metrics_across_seeds.csv
  - paper_table_th{threshold}.csv

The *_seed_std columns are computed across seed-level test means, not across
individual samples. This is the table to use for reviewer-requested robustness
statistics.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


MODEL_ORDER = [
    "radiounet-small",
    "radiounet-medium",
    "transpathnet-small",
    "transpathnet-base",
]

SAMPLE_MEAN_METRICS = [
    "precision",
    "recall",
    "iou",
    "f1",
    "miss_rate",
    "false_alarm_rate",
    "rmse_dbm",
    "mae_dbm",
    "gt_shadow_ratio",
    "pred_shadow_ratio",
]

MICRO_METRICS = [
    "micro_precision",
    "micro_recall",
    "micro_iou",
    "micro_f1",
]

PAPER_METRICS = [
    "precision",
    "recall",
    "iou",
    "f1",
    "rmse_dbm",
    "mae_dbm",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--result_root", required=True,
                   help="Root containing seed_* result directories")
    p.add_argument("--summary_glob", default="seed_*/metrics/summary_metrics_fixed_threshold.csv",
                   help="Glob under result_root for seed-level summary CSV files")
    p.add_argument("--output_dir", default=None,
                   help="Output directory. Defaults to <result_root>/aggregate")
    p.add_argument("--ddof", type=int, default=1,
                   help="Delta degrees of freedom for seed std. Use 1 for sample std.")
    return p.parse_args()


def threshold_tag(threshold: float) -> str:
    if float(threshold).is_integer():
        return str(int(threshold))
    return str(threshold).replace(".", "p").replace("-", "m")


def infer_seed(path: Path) -> int:
    for part in reversed(path.parts):
        m = re.fullmatch(r"seed[_-]?(-?\d+)", part)
        if m:
            return int(m.group(1))
    return -1


def to_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def to_int(value: Any) -> int:
    try:
        if value is None or value == "":
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def mean_std(values: Iterable[float], ddof: int) -> Tuple[float, float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return math.nan, math.nan
    mean = sum(vals) / len(vals)
    if len(vals) <= ddof:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - ddof)
    return mean, math.sqrt(var)


def read_seed_summaries(result_root: Path, summary_glob: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    paths = sorted(result_root.glob(summary_glob))
    if not paths:
        raise FileNotFoundError(
            f"No summary files matched: {result_root / summary_glob}"
        )

    for path in paths:
        inferred_seed = infer_seed(path)
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seed = to_int(row.get("seed", inferred_seed))
                if seed == 0 and not row.get("seed") and inferred_seed != 0:
                    seed = inferred_seed
                row["seed"] = seed
                row["source_csv"] = str(path)
                rows.append(row)

    return rows


def sort_key(row: Dict[str, Any]) -> Tuple[float, int, int, str, int]:
    tag = str(row.get("model_tag", ""))
    model_idx = MODEL_ORDER.index(tag) if tag in MODEL_ORDER else len(MODEL_ORDER)
    return (
        to_float(row.get("threshold_dbm")),
        to_int(row.get("freq_mhz")),
        model_idx,
        tag,
        to_int(row.get("seed")),
    )


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def write_by_seed(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    all_fields = []
    preferred = [
        "seed", "threshold_dbm", "freq_mhz", "model_tag", "model_label",
        "n_samples",
    ]
    for field in preferred:
        if any(field in r for r in rows):
            all_fields.append(field)

    for row in rows:
        for field in row:
            if field not in all_fields:
                all_fields.append(field)

    write_csv(output_dir / "summary_metrics_by_seed.csv", rows, all_fields)


def aggregate(rows: List[Dict[str, Any]], ddof: int) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[float, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            to_float(row.get("threshold_dbm")),
            to_int(row.get("freq_mhz")),
            str(row.get("model_tag")),
        )
        groups[key].append(row)

    out: List[Dict[str, Any]] = []
    for (threshold, freq, tag), sub in groups.items():
        sub = sorted(sub, key=lambda r: to_int(r.get("seed")))
        seeds = [to_int(r.get("seed")) for r in sub]
        n_samples = [to_int(r.get("n_samples")) for r in sub]
        item: Dict[str, Any] = {
            "threshold_dbm": threshold,
            "freq_mhz": freq,
            "model_tag": tag,
            "model_label": sub[0].get("model_label", tag),
            "n_seeds": len(set(seeds)),
            "seeds": " ".join(str(s) for s in sorted(set(seeds))),
            "n_samples_mean": sum(n_samples) / len(n_samples) if n_samples else math.nan,
            "n_samples_min": min(n_samples) if n_samples else math.nan,
            "n_samples_max": max(n_samples) if n_samples else math.nan,
        }

        for metric in SAMPLE_MEAN_METRICS:
            vals = [to_float(r.get(f"{metric}_mean")) for r in sub]
            mean, std = mean_std(vals, ddof=ddof)
            item[f"{metric}_seed_mean"] = mean
            item[f"{metric}_seed_std"] = std

        for metric in MICRO_METRICS:
            vals = [to_float(r.get(metric)) for r in sub]
            mean, std = mean_std(vals, ddof=ddof)
            item[f"{metric}_seed_mean"] = mean
            item[f"{metric}_seed_std"] = std

        out.append(item)

    return sorted(out, key=sort_key)


def format_mean_std(row: Dict[str, Any], metric: str) -> str:
    mean = to_float(row.get(f"{metric}_seed_mean"))
    std = to_float(row.get(f"{metric}_seed_std"))
    if not math.isfinite(mean):
        return "N/A"
    if not math.isfinite(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} +/- {std:.3f}"


def write_paper_tables(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    thresholds = sorted({to_float(r.get("threshold_dbm")) for r in rows})
    for threshold in thresholds:
        if not math.isfinite(threshold):
            continue
        sub = [r for r in rows if abs(to_float(r.get("threshold_dbm")) - threshold) < 1e-9]
        table_rows = []
        for row in sub:
            item = {
                "freq_mhz": row.get("freq_mhz"),
                "model": row.get("model_label"),
                "n_seeds": row.get("n_seeds"),
                "seeds": row.get("seeds"),
            }
            for metric in PAPER_METRICS:
                item[metric] = format_mean_std(row, metric)
            table_rows.append(item)

        fields = ["freq_mhz", "model", "n_seeds", "seeds", *PAPER_METRICS]
        path = output_dir / f"paper_table_th{threshold_tag(threshold)}.csv"
        write_csv(path, table_rows, fields)


def main() -> None:
    cfg = parse_args()
    result_root = Path(cfg.result_root).resolve()
    output_dir = Path(cfg.output_dir).resolve() if cfg.output_dir else result_root / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)

    by_seed = sorted(
        read_seed_summaries(result_root, cfg.summary_glob),
        key=sort_key,
    )
    write_by_seed(by_seed, output_dir)

    aggregated = aggregate(by_seed, ddof=cfg.ddof)
    aggregate_fields = [
        "threshold_dbm", "freq_mhz", "model_tag", "model_label",
        "n_seeds", "seeds", "n_samples_mean", "n_samples_min", "n_samples_max",
    ]
    for metric in SAMPLE_MEAN_METRICS:
        aggregate_fields += [f"{metric}_seed_mean", f"{metric}_seed_std"]
    for metric in MICRO_METRICS:
        aggregate_fields += [f"{metric}_seed_mean", f"{metric}_seed_std"]

    write_csv(output_dir / "summary_metrics_across_seeds.csv", aggregated, aggregate_fields)
    write_paper_tables(aggregated, output_dir)

    readme = output_dir / "README_seed_aggregation.txt"
    readme.write_text(
        "\n".join([
            "[aggregate_seed_metrics]",
            f"- result_root: {result_root}",
            f"- summary_glob: {cfg.summary_glob}",
            f"- ddof for seed std: {cfg.ddof}",
            "- *_seed_mean and *_seed_std are computed across seed-level test means.",
            "- These are not per-sample standard deviations.",
            "- Use paper_table_th*.csv for manuscript tables.",
        ]),
        encoding="utf-8",
    )
    print(f"Saved: {readme}")


if __name__ == "__main__":
    main()
