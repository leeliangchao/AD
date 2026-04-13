"""Run the budget-aligned paper baseline matrix and print the summary payload."""

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
    """Run a budgeted paper baseline matrix config."""

    parser = argparse.ArgumentParser(description="Run the budgeted paper baseline matrix.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "ablation" / "paper_baseline_matrix_v2_budgeted.yaml"),
        help="Path to the budgeted paper baseline matrix YAML file.",
    )
    args = parser.parse_args()

    results = AblationRunner(args.config, output_root=PROJECT_ROOT / "outputs").run()
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
