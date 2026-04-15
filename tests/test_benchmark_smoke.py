"""Smoke tests for benchmark execution and export scripts."""

import json
from pathlib import Path
import subprocess
import sys

from adrf.benchmark.runner import BenchmarkRunner


def test_benchmark_runner_executes_multiple_experiments(tmp_path: Path) -> None:
    """BenchmarkRunner should execute a tiny suite and write suite summaries."""

    project_root = Path(__file__).resolve().parents[1]
    suite_path = tmp_path / "tiny_suite.yaml"
    suite_path.write_text(
        "\n".join(
            [
                "name: tiny_suite",
                "continue_on_error: true",
                "experiments:",
                f"  - {project_root / 'configs' / 'experiment' / 'recon_baseline.yaml'}",
                f"  - {project_root / 'configs' / 'experiment' / 'reference_baseline.yaml'}",
            ]
        ),
        encoding="utf-8",
    )

    results = BenchmarkRunner(suite_path, output_root=tmp_path / "outputs").run()
    suite_dir = Path(results["suite_dir"])

    assert len(results["experiments"]) == 2
    assert all(record["status"] == "completed" for record in results["experiments"])
    assert (suite_dir / "suite_results.json").exists()
    assert (suite_dir / "benchmark_summary.md").exists()
    assert (suite_dir / "benchmark_summary.csv").exists()


def test_benchmark_and_report_scripts_execute() -> None:
    """Benchmark and report export scripts should run end-to-end."""

    project_root = Path(__file__).resolve().parents[1]
    benchmark_result = subprocess.run(
        [sys.executable, "scripts/run_benchmark_suite.py"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert benchmark_result.returncode == 0, benchmark_result.stderr

    payload = json.loads(benchmark_result.stdout)
    suite_dir = payload["suite_dir"]
    run_path = next(
        record["run_path"]
        for record in payload["experiments"]
        if record["status"] == "completed" and record["run_path"]
    )

    report_result = subprocess.run(
        [sys.executable, "scripts/export_experiment_report.py", run_path],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert report_result.returncode == 0, report_result.stderr
    assert report_result.stdout.strip().endswith("report.md")

    summary_result = subprocess.run(
        [sys.executable, "scripts/export_benchmark_summary.py", suite_dir],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert summary_result.returncode == 0, summary_result.stderr
    summary_payload = json.loads(summary_result.stdout)
    assert summary_payload["markdown"].endswith("benchmark_summary.md")


def test_benchmark_runner_reports_invalid_experiment_config_cleanly(tmp_path: Path) -> None:
    suite_path = tmp_path / "tiny_suite.yaml"
    invalid_experiment = tmp_path / "invalid_experiment.yaml"
    invalid_experiment.write_text(
        "- not-a-mapping\n",
        encoding="utf-8",
    )
    suite_path.write_text(
        "\n".join(
            [
                "name: invalid_suite",
                "continue_on_error: true",
                f"experiments:\n  - {invalid_experiment}",
            ]
        ),
        encoding="utf-8",
    )

    results = BenchmarkRunner(suite_path, output_root=tmp_path / "outputs").run()

    record = results["experiments"][0]
    assert record["status"] == "failed"
    assert "must load to a mapping" in record["error"]
