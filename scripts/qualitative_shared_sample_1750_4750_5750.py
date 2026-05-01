#!/usr/bin/env python3
"""
qualitative_shared_sample_1750_4750_5750.py

Generate qualitative figures using ONE shared sample id across representative
frequencies, e.g. 1750 / 4750 / 5750 MHz.

Outputs
-------
1) selected_shared_sample.csv
2) error_map_4models_{freq}MHz_sample{sample}_th{T}.png
3) shadow_detection_4models_{freq}MHz_sample{sample}_th{T}.png
4) error_map_allfreqs_sample{sample}_th{T}.png
5) shadow_detection_allfreqs_sample{sample}_th{T}.png

Shadow visualization colors
---------------------------
- White : normal / TN / background
- Green : TP, correctly detected shadow
- Red   : FP, false alarm / over-detected shadow
- Blue  : FN, missed shadow
- Gray  : building / excluded pixels

Recommended usage
-----------------
python qualitative_shared_sample_1750_4750_5750.py \
    --root /home/james4510/OutdoorRadioMapPrediction/data \
    --ckpt_dir ./model_checkpoints \
    --metrics_csv ./results_1750_4750_5750/metrics/per_sample_metrics_fixed_threshold.csv \
    --result_dir ./results_1750_4750_5750/qualitative \
    --freqs 1750 4750 5750 \
    --sample_id auto \
    --selection_threshold -65 \
    --shadow_threshold -65
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "model"))

from bratlab_dataset import get_dataloaders

try:
    from model.radiounet import build_radiounet
    from model.transpathnet_outdoor import build_transpathnet
except Exception:
    from radiounet import build_radiounet
    from transpathnet_outdoor import build_transpathnet


MODEL_ORDER = [
    "radiounet-small",
    "transpathnet-small",
    "radiounet-medium",
    "transpathnet-base",
]

MODELS_CONFIG = {
    "radiounet-small": {
        "arch": "radiounet",
        "size": "small",
        "label": "RadioUNet-S (35M)",
    },
    "radiounet-medium": {
        "arch": "radiounet",
        "size": "medium",
        "label": "RadioUNet-M (97M)",
    },
    "transpathnet-small": {
        "arch": "transpathnet",
        "size": "small",
        "label": "TransPathNet-S (38M)",
    },
    "transpathnet-base": {
        "arch": "transpathnet",
        "size": "base",
        "label": "TransPathNet-B (111M)",
    },
}

# Display order as 2x2:
DISPLAY_GRID = [
    ["radiounet-small", "transpathnet-small"],
    ["radiounet-medium", "transpathnet-base"],
]

COLOR_TP = np.array([0.00, 0.70, 0.15, 1.00], dtype=np.float32)      # green
COLOR_FP = np.array([1.00, 0.15, 0.15, 1.00], dtype=np.float32)      # red
COLOR_FN = np.array([0.10, 0.25, 1.00, 1.00], dtype=np.float32)      # blue
COLOR_BUILDING = np.array([0.84, 0.84, 0.84, 1.00], dtype=np.float32)
COLOR_BG = np.array([1.00, 1.00, 1.00, 1.00], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--ckpt_dir", default="./model_checkpoints")
    p.add_argument("--metrics_csv", required=True)
    p.add_argument("--result_dir", default="./results_1750_4750_5750/qualitative")

    p.add_argument("--freqs", type=int, nargs="+", default=[1750, 4750, 5750])
    p.add_argument("--sample_id", default="auto",
                   help='Use "auto" to select a shared sample, or provide a numeric sample id.')
    p.add_argument("--selection_threshold", type=float, default=-65.0,
                   help="Threshold used only for selecting the shared sample from metrics_csv.")
    p.add_argument("--shadow_threshold", type=float, default=-65.0,
                   help="Threshold used for qualitative TP/FP/FN visualization.")

    p.add_argument("--selection_criterion",
                   choices=["rm_minus_ts_recall_gap", "small_pair_recall_gap", "average_model_gap"],
                   default="rm_minus_ts_recall_gap",
                   help="Higher score is selected among samples shared by all requested frequencies.")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--min_dbm", type=float, default=-90.0)
    p.add_argument("--max_dbm", type=float, default=0.0)

    p.add_argument("--error_vmax", type=float, default=20.0)
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def denorm(x: np.ndarray, min_dbm: float = -90.0, max_dbm: float = 0.0) -> np.ndarray:
    return x * (max_dbm - min_dbm) + min_dbm


def threshold_tag(threshold: float) -> str:
    threshold = float(threshold)
    if threshold.is_integer():
        return str(int(threshold))
    return str(threshold).replace(".", "p").replace("-", "m")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def filter_metric_rows(rows: List[Dict[str, str]], freqs: List[int],
                       threshold: float, tol: float = 1e-6) -> List[Dict[str, str]]:
    out = []
    for r in rows:
        if int(float(r["freq_mhz"])) not in freqs:
            continue
        if abs(float(r["threshold_dbm"]) - threshold) > tol:
            continue
        if r["model_tag"] not in MODEL_ORDER:
            continue
        out.append(r)
    if not out:
        existing = sorted({float(r["threshold_dbm"]) for r in rows if "threshold_dbm" in r})
        raise RuntimeError(
            f"No rows found for threshold={threshold:.1f} dBm and freqs={freqs}. "
            f"Existing thresholds: {existing}"
        )
    return out


def group_metric_rows(rows: List[Dict[str, str]]) -> Dict[Tuple[int, int], Dict[str, Dict[str, float]]]:
    grouped: Dict[Tuple[int, int], Dict[str, Dict[str, float]]] = {}
    for r in rows:
        freq = int(float(r["freq_mhz"]))
        sample_id = int(float(r["sample_id"]))
        tag = r["model_tag"]
        TP = int(float(r["TP"]))
        FP = int(float(r["FP"]))
        FN = int(float(r["FN"]))
        TN = int(float(r["TN"]))

        grouped.setdefault((freq, sample_id), {})[tag] = {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "precision": float(r["precision"]),
            "recall": float(r["recall"]),
            "iou": float(r["iou"]),
            "f1": float(r["f1"]),
            "miss_rate": float(r.get("miss_rate", FN / (TP + FN + 1e-8))),
            "false_alarm_rate": float(r.get("false_alarm_rate", FP / (FP + TN + 1e-8))),
            "rmse_dbm": float(r.get("rmse_dbm", "nan")),
        }
    return grouped


def score_sample(metrics_by_freq: Dict[int, Dict[str, Dict[str, float]]],
                 criterion: str) -> float:
    scores = []

    for freq, m in metrics_by_freq.items():
        if criterion == "rm_minus_ts_recall_gap":
            scores.append(m["radiounet-medium"]["recall"] - m["transpathnet-small"]["recall"])
        elif criterion == "small_pair_recall_gap":
            scores.append(m["radiounet-small"]["recall"] - m["transpathnet-small"]["recall"])
        elif criterion == "average_model_gap":
            cnn = 0.5 * (m["radiounet-small"]["recall"] + m["radiounet-medium"]["recall"])
            vit = 0.5 * (m["transpathnet-small"]["recall"] + m["transpathnet-base"]["recall"])
            scores.append(cnn - vit)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    return float(np.mean(scores))


def choose_shared_sample(metrics_csv: Path, freqs: List[int], threshold: float,
                         sample_id_arg: str, criterion: str, out_dir: Path) -> int:
    rows = read_csv_rows(metrics_csv)
    rows = filter_metric_rows(rows, freqs, threshold)
    grouped = group_metric_rows(rows)

    # Build sample candidates that exist at all requested frequencies with all four models.
    sample_ids = sorted({sample_id for (_, sample_id) in grouped.keys()})
    candidates = []

    for sid in sample_ids:
        metrics_by_freq = {}
        ok = True
        for f in freqs:
            m = grouped.get((f, sid), {})
            if not all(tag in m for tag in MODEL_ORDER):
                ok = False
                break
            metrics_by_freq[f] = m
        if not ok:
            continue

        score = score_sample(metrics_by_freq, criterion)
        candidates.append((score, sid, metrics_by_freq))

    if not candidates:
        raise RuntimeError(
            f"No common sample id found across freqs={freqs} with all four model rows. "
            "Check the fixed test split and metrics_csv."
        )

    forced_sample: Optional[int] = None
    if str(sample_id_arg).lower() != "auto":
        try:
            forced_sample = int(sample_id_arg)
        except ValueError:
            forced_sample = None

    if forced_sample is not None:
        for score, sid, metrics_by_freq in candidates:
            if sid == forced_sample:
                selected = (score, sid, metrics_by_freq)
                break
        else:
            print(
                f"[WARN] Requested sample_id={forced_sample} is not common across {freqs}. "
                "Falling back to automatic selection."
            )
            selected = max(candidates, key=lambda x: x[0])
    else:
        selected = max(candidates, key=lambda x: x[0])

    score, sid, metrics_by_freq = selected

    # Save top candidates for traceability.
    candidate_rows = []
    for rank, (s, candidate_sid, mbf) in enumerate(sorted(candidates, key=lambda x: x[0], reverse=True)[:30], start=1):
        row = {
            "rank": rank,
            "sample_id": candidate_sid,
            "score": s,
            "criterion": criterion,
            "selection_threshold_dbm": threshold,
        }
        for f in freqs:
            row[f"{f}_RadioUNetM_recall"] = mbf[f]["radiounet-medium"]["recall"]
            row[f"{f}_TransPathNetS_recall"] = mbf[f]["transpathnet-small"]["recall"]
            row[f"{f}_gap_RM_TS"] = mbf[f]["radiounet-medium"]["recall"] - mbf[f]["transpathnet-small"]["recall"]
        candidate_rows.append(row)

    fields = list(candidate_rows[0].keys())
    save_csv(out_dir / "selected_shared_sample_candidates.csv", candidate_rows, fields)

    selected_rows = []
    for f in freqs:
        m = metrics_by_freq[f]
        row = {
            "sample_id": sid,
            "freq_mhz": f,
            "score": score,
            "criterion": criterion,
            "selection_threshold_dbm": threshold,
        }
        for tag in MODEL_ORDER:
            row[f"{tag}_precision"] = m[tag]["precision"]
            row[f"{tag}_recall"] = m[tag]["recall"]
            row[f"{tag}_iou"] = m[tag]["iou"]
            row[f"{tag}_f1"] = m[tag]["f1"]
            row[f"{tag}_TP"] = m[tag]["TP"]
            row[f"{tag}_FP"] = m[tag]["FP"]
            row[f"{tag}_FN"] = m[tag]["FN"]
        selected_rows.append(row)

    save_csv(out_dir / "selected_shared_sample.csv", selected_rows, list(selected_rows[0].keys()))

    print(f"[Sample Selection] selected sample_id={sid} | score={score:.4f} | criterion={criterion}")
    return int(sid)


def resolve_ckpt_path(ckpt_dir: str, tag: str, freq: int) -> Path:
    root = Path(ckpt_dir)

    # Existing user preference: use RadioUNet-S 4750 v2 when available.
    if tag == "radiounet-small" and freq == 4750:
        v2 = root / "checkpoints_radiounet_small_4750_v2" / "best.pth"
        if v2.exists():
            return v2

    return root / f"checkpoints_{tag.replace('-', '_')}_{freq}" / "best.pth"


def build_model(tag: str):
    cfg = MODELS_CONFIG[tag]
    if cfg["arch"] == "radiounet":
        return build_radiounet(cfg["size"])
    if cfg["arch"] == "transpathnet":
        return build_transpathnet(cfg["size"])
    raise ValueError(f"Unknown arch: {cfg['arch']}")


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


def load_models(freq: int, ckpt_dir: str, device: torch.device) -> Dict[str, torch.nn.Module]:
    models = {}
    for tag in MODEL_ORDER:
        path = resolve_ckpt_path(ckpt_dir, tag, freq)
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
        models[tag] = load_model(tag, path, device)
    return models


def sample_number_from_batch(n) -> int:
    if isinstance(n, torch.Tensor):
        return int(n[0].item()) if n.ndim > 0 else int(n.item())
    if isinstance(n, (list, tuple)):
        first = n[0]
        if isinstance(first, torch.Tensor):
            return int(first.item())
        return int(first)
    return int(n)


def find_sample(root: str, freq: int, sample_id: int, img_size: int, num_workers: int):
    _, _, te_loader = get_dataloaders(
        root=root,
        freq=freq,
        img_size=img_size,
        batch_size=1,
        num_workers=num_workers,
    )

    for x, y, n in te_loader:
        cur = sample_number_from_batch(n)
        if cur == sample_id:
            return x, y, cur

    raise RuntimeError(f"Sample id {sample_id} not found in fixed test loader for {freq} MHz.")


def predict_models(models: Dict[str, torch.nn.Module], x: torch.Tensor, device: torch.device,
                   min_dbm: float, max_dbm: float) -> Dict[str, np.ndarray]:
    x_gpu = x.to(device)
    preds = {}
    with torch.no_grad():
        for tag, model in models.items():
            out = model(x_gpu)
            if isinstance(out, (tuple, list)):
                fine = out[1] if len(out) >= 2 else out[0]
            else:
                fine = out
            pred = fine[0, 0].detach().cpu().numpy()
            preds[tag] = denorm(pred, min_dbm, max_dbm)
    return preds


def compute_confusion_masks(pred_dbm: np.ndarray, gt_dbm: np.ndarray,
                            building: np.ndarray, threshold: float):
    valid = building < 0.5
    pred_shadow = (pred_dbm < threshold) & valid
    gt_shadow = (gt_dbm < threshold) & valid

    TP = pred_shadow & gt_shadow
    FP = pred_shadow & (~gt_shadow)
    FN = (~pred_shadow) & gt_shadow
    TN = (~pred_shadow) & (~gt_shadow) & valid
    return TP, FP, FN, TN


def metrics_from_masks(TP, FP, FN, TN) -> Dict[str, float]:
    tp = int(TP.sum())
    fp = int(FP.sum())
    fn = int(FN.sum())
    tn = int(TN.sum())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "f1": f1,
    }


def shadow_rgba(TP, FP, FN, building) -> np.ndarray:
    h, w = building.shape
    img = np.zeros((h, w, 4), dtype=np.float32)
    img[:] = COLOR_BG
    img[building > 0.5] = COLOR_BUILDING
    img[TP] = COLOR_TP
    img[FP] = COLOR_FP
    img[FN] = COLOR_FN
    return img


def error_cmap():
    return LinearSegmentedColormap.from_list(
        "white_to_red",
        [(1.0, 1.0, 1.0), (1.0, 0.80, 0.70), (1.0, 0.30, 0.20), (0.45, 0.0, 0.0)],
        N=256,
    )


def plot_error_maps_for_freq(freq: int, sample_id: int, gt_dbm: np.ndarray,
                             building: np.ndarray, preds: Dict[str, np.ndarray],
                             out_dir: Path, threshold: float, vmax: float, dpi: int):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    cmap = error_cmap()

    last_im = None
    for r, row in enumerate(DISPLAY_GRID):
        for c, tag in enumerate(row):
            ax = axes[r, c]
            err = np.abs(preds[tag] - gt_dbm)
            display = np.ma.masked_where(building > 0.5, err)
            cmap_local = cmap.copy()
            cmap_local.set_bad((0.84, 0.84, 0.84, 1.0))
            last_im = ax.imshow(display, cmap=cmap_local, vmin=0, vmax=vmax, origin="lower", interpolation="nearest")
            ax.set_title(
                f"{MODELS_CONFIG[tag]['label']}\nMean: {np.nanmean(display):.2f} dB | Max: {np.nanmax(display):.2f} dB",
                fontsize=12,
                fontweight="bold",
            )
            ax.axis("off")

    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.86, pad=0.02)
    cbar.set_label("Absolute RSS Error (dB)", fontsize=11)

    fig.suptitle(
        f"Error Map Comparison | Sample {sample_id} | {freq} MHz",
        fontsize=17,
        fontweight="bold",
    )
    path = out_dir / f"error_map_4models_{freq}MHz_sample{sample_id}_th{threshold_tag(threshold)}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_shadow_maps_for_freq(freq: int, sample_id: int, gt_dbm: np.ndarray,
                              building: np.ndarray, preds: Dict[str, np.ndarray],
                              out_dir: Path, threshold: float, dpi: int):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for r, row in enumerate(DISPLAY_GRID):
        for c, tag in enumerate(row):
            ax = axes[r, c]
            TP, FP, FN, TN = compute_confusion_masks(preds[tag], gt_dbm, building, threshold)
            m = metrics_from_masks(TP, FP, FN, TN)
            ax.imshow(shadow_rgba(TP, FP, FN, building), origin="lower", interpolation="nearest")
            ax.set_title(
                f"{MODELS_CONFIG[tag]['label']}\n"
                f"TP:{m['TP']:,} FP:{m['FP']:,} FN:{m['FN']:,} | R:{m['recall']:.3f} F1:{m['f1']:.3f}",
                fontsize=12,
                fontweight="bold",
            )
            ax.axis("off")

    handles = [
        mpatches.Patch(color=COLOR_TP, label="TP: Correct shadow"),
        mpatches.Patch(color=COLOR_FP, label="FP: False alarm"),
        mpatches.Patch(color=COLOR_FN, label="FN: Missed shadow"),
        mpatches.Patch(color=COLOR_BUILDING, label="Building / Excluded"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=11, frameon=True)

    fig.suptitle(
        f"Shadow Region Detection Comparison | Sample {sample_id} | {freq} MHz | Threshold {threshold:.1f} dBm",
        fontsize=17,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    path = out_dir / f"shadow_detection_4models_{freq}MHz_sample{sample_id}_th{threshold_tag(threshold)}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_allfreqs_error(freq_data: Dict[int, Dict[str, object]], sample_id: int,
                        out_dir: Path, threshold: float, vmax: float, dpi: int):
    freqs = list(freq_data.keys())
    fig, axes = plt.subplots(len(freqs), len(MODEL_ORDER), figsize=(4.0 * len(MODEL_ORDER), 3.7 * len(freqs)))
    if len(freqs) == 1:
        axes = np.expand_dims(axes, axis=0)

    cmap = error_cmap()
    last_im = None

    for i, freq in enumerate(freqs):
        gt_dbm = freq_data[freq]["gt_dbm"]
        building = freq_data[freq]["building"]
        preds = freq_data[freq]["preds"]

        for j, tag in enumerate(MODEL_ORDER):
            ax = axes[i, j]
            err = np.abs(preds[tag] - gt_dbm)
            display = np.ma.masked_where(building > 0.5, err)
            cmap_local = cmap.copy()
            cmap_local.set_bad((0.84, 0.84, 0.84, 1.0))
            last_im = ax.imshow(display, cmap=cmap_local, vmin=0, vmax=vmax, origin="lower", interpolation="nearest")
            ax.set_title(
                f"{freq} MHz | {MODELS_CONFIG[tag]['label']}\nMean {np.nanmean(display):.2f} dB",
                fontsize=10,
                fontweight="bold",
            )
            ax.axis("off")

    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.86, pad=0.01)
    cbar.set_label("Absolute RSS Error (dB)", fontsize=10)

    fig.suptitle(
        f"Error Map Comparison across Frequencies | Shared Sample {sample_id}",
        fontsize=17,
        fontweight="bold",
    )
    path = out_dir / f"error_map_allfreqs_sample{sample_id}_th{threshold_tag(threshold)}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_allfreqs_shadow(freq_data: Dict[int, Dict[str, object]], sample_id: int,
                         out_dir: Path, threshold: float, dpi: int):
    freqs = list(freq_data.keys())
    fig, axes = plt.subplots(len(freqs), len(MODEL_ORDER), figsize=(4.0 * len(MODEL_ORDER), 3.7 * len(freqs)))
    if len(freqs) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, freq in enumerate(freqs):
        gt_dbm = freq_data[freq]["gt_dbm"]
        building = freq_data[freq]["building"]
        preds = freq_data[freq]["preds"]

        for j, tag in enumerate(MODEL_ORDER):
            ax = axes[i, j]
            TP, FP, FN, TN = compute_confusion_masks(preds[tag], gt_dbm, building, threshold)
            m = metrics_from_masks(TP, FP, FN, TN)
            ax.imshow(shadow_rgba(TP, FP, FN, building), origin="lower", interpolation="nearest")
            ax.set_title(
                f"{freq} MHz | {MODELS_CONFIG[tag]['label']}\n"
                f"FP:{m['FP']:,} FN:{m['FN']:,} R:{m['recall']:.3f}",
                fontsize=10,
                fontweight="bold",
            )
            ax.axis("off")

    handles = [
        mpatches.Patch(color=COLOR_TP, label="TP: Correct shadow"),
        mpatches.Patch(color=COLOR_FP, label="FP: False alarm"),
        mpatches.Patch(color=COLOR_FN, label="FN: Missed shadow"),
        mpatches.Patch(color=COLOR_BUILDING, label="Building / Excluded"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=11, frameon=True)
    fig.suptitle(
        f"Shadow Region Detection across Frequencies | Shared Sample {sample_id} | T={threshold:.1f} dBm",
        fontsize=17,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    path = out_dir / f"shadow_detection_allfreqs_sample{sample_id}_th{threshold_tag(threshold)}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    cfg = parse_args()
    out_dir = Path(cfg.result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = Path(cfg.metrics_csv)
    if not metrics_csv.exists():
        raise FileNotFoundError(metrics_csv)

    selected_sample_id = choose_shared_sample(
        metrics_csv=metrics_csv,
        freqs=cfg.freqs,
        threshold=cfg.selection_threshold,
        sample_id_arg=cfg.sample_id,
        criterion=cfg.selection_criterion,
        out_dir=out_dir,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    freq_data: Dict[int, Dict[str, object]] = {}

    for freq in cfg.freqs:
        print("\n" + "=" * 80)
        print(f"Qualitative generation | {freq} MHz | sample_id={selected_sample_id}")
        print("=" * 80)

        x, y, sample_n = find_sample(cfg.root, freq, selected_sample_id, cfg.img_size, cfg.num_workers)
        building = x[0, 0].detach().cpu().numpy()
        gt_dbm = denorm(y[0, 0].detach().cpu().numpy(), cfg.min_dbm, cfg.max_dbm)

        models = load_models(freq, cfg.ckpt_dir, device)
        preds = predict_models(models, x, device, cfg.min_dbm, cfg.max_dbm)

        freq_data[freq] = {
            "building": building,
            "gt_dbm": gt_dbm,
            "preds": preds,
        }

        plot_error_maps_for_freq(
            freq=freq,
            sample_id=selected_sample_id,
            gt_dbm=gt_dbm,
            building=building,
            preds=preds,
            out_dir=out_dir,
            threshold=cfg.shadow_threshold,
            vmax=cfg.error_vmax,
            dpi=cfg.dpi,
        )

        plot_shadow_maps_for_freq(
            freq=freq,
            sample_id=selected_sample_id,
            gt_dbm=gt_dbm,
            building=building,
            preds=preds,
            out_dir=out_dir,
            threshold=cfg.shadow_threshold,
            dpi=cfg.dpi,
        )

    plot_allfreqs_error(
        freq_data=freq_data,
        sample_id=selected_sample_id,
        out_dir=out_dir,
        threshold=cfg.shadow_threshold,
        vmax=cfg.error_vmax,
        dpi=cfg.dpi,
    )

    plot_allfreqs_shadow(
        freq_data=freq_data,
        sample_id=selected_sample_id,
        out_dir=out_dir,
        threshold=cfg.shadow_threshold,
        dpi=cfg.dpi,
    )

    print("\nAll qualitative figures generated.")
    print(f"Selected shared sample id: {selected_sample_id}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
