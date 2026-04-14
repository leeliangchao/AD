"""Print the current official baseline comparison contract."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from adrf.utils.config import load_yaml_config


OFFICIAL_CONTRACT = PROJECT_ROOT / "configs" / "ablation" / "paper_baseline_matrix_official_v1.yaml"
FIXED_METRICS = ("image_auroc", "pixel_auroc", "pixel_aupr", "train_time", "total_time")


def _core_baselines(config: dict[str, object]) -> list[str]:
    compatibility = config.get("compatibility", {})
    if not isinstance(compatibility, dict):
        return []

    combos: list[str] = []
    for normality in config.get("normality", []):
        evidence_items = compatibility.get(normality, [])
        if not isinstance(evidence_items, list):
            continue
        for evidence in evidence_items:
            combos.append(f"{normality} + {evidence}")
    return combos


def main() -> int:
    config = load_yaml_config(OFFICIAL_CONTRACT)
    datasets = config.get("datasets", [])
    overrides = config.get("overrides", {})
    normality_overrides = overrides.get("normality", {}) if isinstance(overrides, dict) else {}

    print(f"official_contract: {OFFICIAL_CONTRACT}")
    print(f"claim_bearing_contract: {OFFICIAL_CONTRACT.relative_to(PROJECT_ROOT)}")
    print("datasets:")
    for dataset in datasets:
        if not isinstance(dataset, dict):
            continue
        print(f"- {dataset.get('name')} -> {dataset.get('dataset_config')}")
    print(f"seeds: {config.get('seeds')}")
    print(f"runtime_config: {config.get('runtime_config')}")
    print(f"backend: {normality_overrides.get('backend')}")
    print(f"metrics: {', '.join(FIXED_METRICS)}")
    print("core_baselines:")
    for combo in _core_baselines(config):
        print(f"- {combo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
