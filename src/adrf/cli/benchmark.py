"""CLI entrypoint for running benchmark suites."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from adrf.benchmark.runner import BenchmarkRunner


def main(argv: Sequence[str] | None = None) -> int:
    """Run one benchmark suite and print its summary metadata."""

    parser = argparse.ArgumentParser(description="Run a benchmark suite.")
    parser.add_argument(
        "suite",
        nargs="?",
        default=Path("configs/benchmark/baseline_suite.yaml"),
        type=Path,
        help="Path to the benchmark suite YAML file.",
    )
    args = parser.parse_args(argv)

    results = BenchmarkRunner(args.suite).run()
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
