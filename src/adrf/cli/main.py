"""CLI entrypoint for running a single experiment config."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from adrf.runner.experiment_runner import ExperimentRunner


def main(argv: Sequence[str] | None = None) -> int:
    """Run one experiment config and print the resulting metrics."""

    parser = argparse.ArgumentParser(description="Run one AD research experiment.")
    parser.add_argument(
        "config",
        nargs="?",
        default=Path("configs/experiment/feature_baseline.yaml"),
        type=Path,
        help="Path to the experiment config file.",
    )
    args = parser.parse_args(argv)

    results = ExperimentRunner(args.config).run()
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
