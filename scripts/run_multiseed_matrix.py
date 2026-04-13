"""Run a multi-seed ablation matrix and print aggregated results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from adrf.ablation.runner import AblationRunner


def main() -> int:
    """Run a multi-seed ablation matrix config and print the resulting summary."""

    parser = argparse.ArgumentParser(description="Run a multi-seed ablation matrix.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "ablation" / "diffusion_evidence_multiseed.yaml"),
        help="Path to the multi-seed ablation matrix YAML file.",
    )
    args = parser.parse_args()

    results = AblationRunner(args.config, output_root=PROJECT_ROOT / "outputs").run()
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

