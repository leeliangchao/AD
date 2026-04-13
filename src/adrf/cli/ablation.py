"""CLI entrypoint for running ablation matrices."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from adrf.ablation.runner import AblationRunner


def main(argv: Sequence[str] | None = None) -> int:
    """Run one ablation matrix and print its summary metadata."""

    parser = argparse.ArgumentParser(description="Run an ablation matrix.")
    parser.add_argument(
        "config",
        nargs="?",
        default=Path("configs/ablation/paper_baseline_matrix_v3_audited.yaml"),
        type=Path,
        help="Path to the ablation matrix YAML file.",
    )
    args = parser.parse_args(argv)

    results = AblationRunner(args.config).run()
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
