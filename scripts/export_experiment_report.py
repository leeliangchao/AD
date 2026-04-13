"""Export a markdown report for one experiment run."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from adrf.reporting.report import export_experiment_report, find_latest_run_dir


def main() -> int:
    """Export a report for the latest run or an explicitly provided run directory."""

    run_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else find_latest_run_dir(PROJECT_ROOT / "outputs" / "runs")
    report_path = export_experiment_report(run_dir)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

