"""Run the reconstruction baseline experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from adrf.runner.experiment_runner import ExperimentRunner


def main() -> int:
    """Run the default reconstruction baseline config and print the results."""

    config_path = PROJECT_ROOT / "configs" / "experiment" / "recon_baseline.yaml"
    results = ExperimentRunner(config_path).run()
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

