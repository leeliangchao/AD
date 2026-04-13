"""CLI entrypoint for exporting experiment and benchmark reports."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from adrf.reporting.report import export_experiment_report, find_latest_run_dir
from adrf.reporting.summary import (
    export_ablation_summary,
    export_benchmark_summary,
    find_latest_ablation_dir,
    find_latest_benchmark_dir,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Export experiment, benchmark, or ablation report assets."""

    parser = argparse.ArgumentParser(description="Export experiment, benchmark, or ablation reports.")
    parser.add_argument(
        "--kind",
        choices=("experiment", "benchmark", "ablation"),
        default="experiment",
        help="Which type of report to export.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Optional run or benchmark directory. Defaults to the latest matching directory.",
    )
    args = parser.parse_args(argv)

    if args.kind == "experiment":
        run_dir = args.path or find_latest_run_dir()
        output = export_experiment_report(run_dir)
        print(output)
    elif args.kind == "benchmark":
        suite_dir = args.path or find_latest_benchmark_dir()
        outputs = export_benchmark_summary(suite_dir)
        print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    else:
        matrix_dir = args.path or find_latest_ablation_dir()
        outputs = export_ablation_summary(matrix_dir)
        print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
