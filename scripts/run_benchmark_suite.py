"""Run a benchmark suite and export its summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from adrf.benchmark.runner import BenchmarkRunner


def main() -> int:
    """Run the default benchmark suite or a user-provided suite config."""

    suite_path = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) > 1
        else PROJECT_ROOT / "configs" / "benchmark" / "baseline_suite.yaml"
    )
    results = BenchmarkRunner(suite_path, output_root=PROJECT_ROOT / "outputs").run()
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

