#!/usr/bin/env python3
"""
check_setup.py

Check the runtime environment, dataset and checkpoints before running the
training / evaluation scripts.

Usage
-----
python scripts/check_setup.py
python scripts/check_setup.py --root /path/to/data --ckpt_dir /path/to/model_checkpoints

The result is OK when the Python packages and the dataset are found.
Checkpoints are only reported, because they are needed for evaluation but not
for training.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

FREQS = [1750, 2750, 3750, 4750, 5750]
MODEL_TAGS = ["radiounet_small", "radiounet_medium", "transpathnet_small", "transpathnet_base"]
# import name -> distribution name
PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "PIL": "pillow",
    "pytorch_msssim": "pytorch-msssim",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(REPO_ROOT / "data"), help="BRAT-Lab dataset root")
    p.add_argument("--ckpt_dir", default=str(REPO_ROOT / "model_checkpoints"))
    p.add_argument("--freqs", type=int, nargs="+", default=FREQS)
    return p.parse_args()


def check_packages() -> bool:
    print("[1/5] Python packages")
    ok = True
    for name, dist in PACKAGES.items():
        try:
            mod = importlib.import_module(name)
        except ImportError as e:
            print(f"  MISSING  {name:<16}{e}")
            ok = False
            continue
        try:
            version = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            version = getattr(mod, "__version__", "")
        print(f"  OK       {name:<16}{version}")
    return ok


def check_gpu() -> None:
    import torch

    print("[2/5] GPU")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  OK       cuda:{i}  {torch.cuda.get_device_name(i)}")
    else:
        print("  WARN     CUDA is not available. Scripts will run on the CPU.")


def check_project_code() -> bool:
    print("[3/5] Project code (ikeee)")
    try:
        from ikeee.bratlab_dataset import get_dataloaders  # noqa: F401
        from ikeee.model.radiounet import build_radiounet  # noqa: F401
        from ikeee.model.transpathnet_outdoor import build_transpathnet  # noqa: F401
    except Exception as e:
        print(f"  FAIL     {type(e).__name__}: {e}")
        return False
    print("  OK       ikeee.bratlab_dataset, ikeee.model")
    return True


def count_files(folder: Path, pattern: str) -> int:
    return sum(1 for _ in folder.glob(pattern)) if folder.is_dir() else 0


def check_dataset(root: Path, freqs: List[int]) -> bool:
    print(f"[4/5] Dataset: {root}")
    if not root.is_dir():
        print("  MISSING  dataset folder")
        return False

    ok = True
    sta = root / "stations_position.txt"
    if sta.is_file():
        n_lines = sum(1 for line in sta.read_text().splitlines() if line.strip())
        print(f"  OK       stations_position.txt ({n_lines} samples)")
    else:
        print("  MISSING  stations_position.txt")
        ok = False

    n_bp = count_files(root / "buildings_position", "buildings_*.mat")
    print(f"  {'OK' if n_bp else 'MISSING':<9}buildings_position ({n_bp} files)")
    ok &= n_bp > 0

    for freq in freqs:
        n_rm = count_files(root / f"receivedpower_{freq}MHz_mat", f"TOTAL_REC_{freq}MHz_*.mat")
        print(f"  {'OK' if n_rm else 'MISSING':<9}receivedpower_{freq}MHz_mat ({n_rm} files)")
        ok &= n_rm > 0
    return ok


def check_checkpoints(ckpt_dir: Path, freqs: List[int]) -> int:
    print(f"[5/5] Checkpoints: {ckpt_dir}")
    print("  " + "model".ljust(22) + "".join(f"{freq:>7}" for freq in freqs))
    found = 0
    for tag in MODEL_TAGS:
        marks = []
        for freq in freqs:
            exists = (ckpt_dir / f"checkpoints_{tag}_{freq}" / "best.pth").is_file()
            found += int(exists)
            marks.append("O" if exists else "-")
        print("  " + tag.ljust(22) + "".join(f"{m:>7}" for m in marks))
    if (ckpt_dir / "checkpoints_radiounet_small_4750_v2" / "best.pth").is_file():
        print("  (checkpoints_radiounet_small_4750_v2 found: used for RadioUNet-S at 4750 MHz)")
    print("  O: best.pth found, -: not found")
    return found


def main() -> None:
    args = parse_args()

    packages_ok = check_packages()
    code_ok = False
    if packages_ok:
        check_gpu()
        code_ok = check_project_code()
    dataset_ok = check_dataset(Path(args.root), args.freqs)
    n_ckpt = check_checkpoints(Path(args.ckpt_dir), args.freqs)

    print()
    if packages_ok and code_ok and dataset_ok:
        print("Result: OK")
        if n_ckpt == 0:
            print("No checkpoints found. Train models first, or copy the checkpoints before evaluation.")
        sys.exit(0)
    print("Result: FAIL (see MISSING / FAIL lines above)")
    sys.exit(1)


if __name__ == "__main__":
    main()
