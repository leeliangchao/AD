"""Export markdown and CSV summaries for a benchmark suite run."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from adrf.reporting.summary import export_benchmark_summary, find_latest_benchmark_dir


def main() -> int:
    """Export benchmark summaries for the latest suite or an explicitly provided suite directory."""

    suite_dir = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) > 1
        else find_latest_benchmark_dir(PROJECT_ROOT / "outputs" / "benchmarks")
    )
    outputs = export_benchmark_summary(suite_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

