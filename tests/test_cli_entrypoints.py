"""Tests for top-level and packaged CLI entrypoints."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from adrf.logging.run_logger import RunLogger


def test_main_py_dispatches_report_subcommand(tmp_path: Path) -> None:
    """The repository entrypoint should dispatch to the report CLI."""

    project_root = Path(__file__).resolve().parents[1]
    logger = RunLogger(base_dir=tmp_path / "runs")
    logger.start_run("demo_main_entrypoint", config={"normality": {"name": "autoencoder"}})
    logger.log_metrics({"evaluation": {"image_auroc": 0.9}})
    logger.finish_run()

    result = subprocess.run(
        [sys.executable, "main.py", "report", "--kind", "experiment", "--path", str(logger.run_dir)],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("report.md")


def test_pyproject_declares_installable_cli_scripts() -> None:
    """The package metadata should expose the supported CLI entrypoints."""

    pyproject_text = Path(__file__).resolve().parents[1].joinpath("pyproject.toml").read_text(encoding="utf-8")

    assert "[project.scripts]" in pyproject_text
    assert 'adrf = "adrf.cli.app:main"' in pyproject_text
    assert 'adrf-report = "adrf.cli.report:main"' in pyproject_text
