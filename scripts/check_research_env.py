"""Print a minimal environment and path checklist for research runs."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "mvtec"
OFFICIAL_CONTRACT = PROJECT_ROOT / "configs" / "ablation" / "paper_baseline_matrix_official_v1.yaml"
OFFICIAL_FILLED_SMOKE = PROJECT_ROOT / "configs" / "ablation" / "paper_baseline_matrix_official_v1_filled_smoke.yaml"
REQUIRED_CATEGORIES = ("bottle", "capsule", "grid")


def _status(path: Path) -> str:
    return "OK" if path.exists() else "MISSING"


def main() -> int:
    python_version = platform.python_version()
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available and torch.cuda.device_count() > 0 else "unavailable"

    print(f"python: {python_version}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {cuda_available}")
    print(f"gpu_name: {gpu_name}")
    print(f"data/mvtec: {_status(DATA_ROOT)} ({DATA_ROOT})")
    for category in REQUIRED_CATEGORIES:
        category_path = DATA_ROOT / category
        print(f"{category}: {_status(category_path)} ({category_path})")
    print(f"official_contract: {_status(OFFICIAL_CONTRACT)} ({OFFICIAL_CONTRACT})")
    print(f"official_filled_smoke: {_status(OFFICIAL_FILLED_SMOKE)} ({OFFICIAL_FILLED_SMOKE})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
