"""Tests for report and summary export helpers."""

import json
from pathlib import Path
import sys

from adrf.logging.run_logger import RunLogger
from adrf.cli.report import main as report_main
from adrf.reporting.report import export_experiment_report
from adrf.reporting.summary import export_ablation_summary, write_benchmark_summary


def test_export_experiment_report_writes_markdown(tmp_path: Path) -> None:
    """Single-experiment report export should create a markdown file."""

    logger = RunLogger(base_dir=tmp_path / "runs")
    logger.start_run("demo_report", config={"normality": {"name": "autoencoder"}})
    logger.log_metrics({"evaluation": {"image_auroc": 0.9}})
    logger.finish_run()

    report_path = export_experiment_report(logger.run_dir)

    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "demo_report" in report_text
    assert "image_auroc" in report_text


def test_export_experiment_report_gracefully_handles_missing_metrics_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "demo_partial"
    run_dir.mkdir(parents=True)
    (run_dir / "run_info.json").write_text(
        json.dumps({"run_name": "demo_partial", "status": "failed"}),
        encoding="utf-8",
    )
    (run_dir / "config_snapshot.yaml").write_text(
        "normality:\n  name: autoencoder\n",
        encoding="utf-8",
    )

    report_path = export_experiment_report(run_dir)

    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "demo_partial" in report_text
    assert "failed" in report_text


def test_export_experiment_report_gracefully_handles_missing_config_snapshot(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "demo_no_config"
    run_dir.mkdir(parents=True)
    (run_dir / "run_info.json").write_text(
        json.dumps({"run_name": "demo_no_config", "status": "failed"}),
        encoding="utf-8",
    )

    report_path = export_experiment_report(run_dir)

    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "demo_no_config" in report_text
    assert "## Config" in report_text


def test_export_experiment_report_gracefully_handles_malformed_metrics_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "demo_bad_metrics"
    run_dir.mkdir(parents=True)
    (run_dir / "run_info.json").write_text(
        json.dumps({"run_name": "demo_bad_metrics", "status": "failed"}),
        encoding="utf-8",
    )
    (run_dir / "config_snapshot.yaml").write_text(
        "normality:\n  name: autoencoder\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text("{not-json", encoding="utf-8")

    report_path = export_experiment_report(run_dir)

    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "demo_bad_metrics" in report_text
    assert "failed" in report_text


def test_write_benchmark_summary_creates_markdown_and_csv(tmp_path: Path) -> None:
    """Benchmark summary export should create markdown, CSV, and JSON assets."""

    outputs = write_benchmark_summary(
        records=[
            {
                "experiment_name": "feature_baseline",
                "status": "completed",
                "metrics": {"image_auroc": 1.0, "pixel_auroc": 0.9, "pixel_aupr": 0.8},
                "run_path": "/tmp/run",
                "config_path": "/tmp/config.yaml",
            }
        ],
        output_dir=tmp_path / "suite",
        suite_name="demo_suite",
    )

    assert outputs["markdown"].exists()
    assert outputs["csv"].exists()
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert payload["suite_name"] == "demo_suite"


def test_write_benchmark_summary_falls_back_to_record_budget_when_run_info_missing(tmp_path: Path) -> None:
    outputs = write_benchmark_summary(
        records=[
            {
                "experiment_name": "feature_baseline",
                "status": "completed",
                "dataset": "mvtec_bottle",
                "representation": "feature",
                "normality": "feature_memory",
                "evidence": "feature_distance",
                "metrics": {"image_auroc": 1.0, "pixel_auroc": 0.9, "pixel_aupr": 0.8, "total_time": 12.0},
                "budget": {"max_epochs": 10, "batch_size": 32, "backend": "legacy"},
                "run_path": "",
                "config_path": "/tmp/config.yaml",
            }
        ],
        output_dir=tmp_path / "suite",
        suite_name="demo_suite",
    )

    markdown = outputs["markdown"].read_text(encoding="utf-8")
    csv_text = outputs["csv"].read_text(encoding="utf-8")

    assert "ep=10, bs=32, backend=legacy" in markdown
    assert "12.0" in markdown
    assert "legacy" in csv_text


def test_export_ablation_summary_creates_markdown_and_csv(tmp_path: Path) -> None:
    """Ablation summary export should create markdown, CSV, and JSON assets."""

    matrix_dir = tmp_path / "ablations" / "demo_matrix"
    matrix_dir.mkdir(parents=True)
    (matrix_dir / "matrix_results.json").write_text(
        json.dumps(
            {
                "matrix_name": "demo_matrix",
                "experiments": [
                    {
                        "experiment_name": "demo_experiment",
                        "status": "completed",
                        "dataset": "mvtec_bottle",
                        "representation": "pixel",
                        "normality": "autoencoder",
                        "evidence": "reconstruction_residual",
                        "metrics": {"image_auroc": 0.9, "pixel_auroc": 0.8, "pixel_aupr": 0.7},
                        "run_path": "",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    outputs = export_ablation_summary(matrix_dir)

    assert outputs["markdown"].exists()
    assert outputs["csv"].exists()
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert payload["matrix_name"] == "demo_matrix"


def test_report_cli_supports_ablation_kind(tmp_path: Path, monkeypatch, capsys) -> None:
    """The report CLI should export ablation summaries via --kind ablation."""

    matrix_dir = tmp_path / "ablations" / "demo_matrix"
    matrix_dir.mkdir(parents=True)
    (matrix_dir / "matrix_results.json").write_text(
        json.dumps({"matrix_name": "demo_matrix", "experiments": []}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["adrf-report", "--kind", "ablation", "--path", str(matrix_dir)],
    )

    exit_code = report_main()
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert "ablation_summary.md" in stdout
