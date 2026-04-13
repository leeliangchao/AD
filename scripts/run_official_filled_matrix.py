"""Run the stronger official baseline matrix and print the summary payload."""

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
    """Run a filled official baseline matrix config."""

    parser = argparse.ArgumentParser(description="Run the stronger official baseline matrix.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "ablation" / "paper_baseline_matrix_official_v1_filled_smoke.yaml"),
        help="Path to the filled official baseline matrix YAML file.",
    )
    args = parser.parse_args()

    results = AblationRunner(args.config, output_root=PROJECT_ROOT / "outputs").run()
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
