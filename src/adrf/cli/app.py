"""Unified CLI entrypoint for the AD research framework."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from adrf.cli import ablation as ablation_cli
from adrf.cli import benchmark as benchmark_cli
from adrf.cli import main as experiment_cli
from adrf.cli import report as report_cli


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""

    parser = argparse.ArgumentParser(description="AD research framework CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    experiment_parser = subparsers.add_parser("experiment", help="Run one experiment config.")
    experiment_parser.add_argument(
        "config",
        nargs="?",
        default="configs/experiment/feature_baseline.yaml",
        help="Path to the experiment config file.",
    )

    benchmark_parser = subparsers.add_parser("benchmark", help="Run one benchmark suite.")
    benchmark_parser.add_argument(
        "suite",
        nargs="?",
        default="configs/benchmark/baseline_suite.yaml",
        help="Path to the benchmark suite YAML file.",
    )

    ablation_parser = subparsers.add_parser("ablation", help="Run one ablation matrix config.")
    ablation_parser.add_argument(
        "config",
        nargs="?",
        default="configs/ablation/paper_baseline_matrix_v3_audited.yaml",
        help="Path to the ablation matrix YAML file.",
    )

    report_parser = subparsers.add_parser("report", help="Export experiment, benchmark, or ablation reports.")
    report_parser.add_argument(
        "--kind",
        choices=("experiment", "benchmark", "ablation"),
        default="experiment",
        help="Which type of report to export.",
    )
    report_parser.add_argument(
        "--path",
        default=None,
        help="Optional run, benchmark, or ablation directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch to the requested subcommand."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "experiment":
        return experiment_cli.main([args.config])
    if args.command == "benchmark":
        return benchmark_cli.main([args.suite])
    if args.command == "ablation":
        return ablation_cli.main([args.config])
    if args.command == "report":
        report_args: list[str] = ["--kind", args.kind]
        if args.path is not None:
            report_args.extend(["--path", args.path])
        return report_cli.main(report_args)
    raise ValueError(f"Unsupported CLI command: {args.command}")
