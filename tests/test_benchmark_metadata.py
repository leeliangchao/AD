"""Tests for benchmark record metadata enrichment."""

from __future__ import annotations

from pathlib import Path

from adrf.benchmark.runner import BenchmarkRunner


def test_benchmark_runner_records_component_metadata(tmp_path: Path) -> None:
    """Benchmark summaries should retain dataset and component names per record."""

    project_root = Path(__file__).resolve().parents[1]
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        "\n".join(
            [
                "name: metadata_suite",
                "continue_on_error: true",
                "experiments:",
                f"  - {project_root / 'configs' / 'experiment' / 'diffusion_baseline.yaml'}",
            ]
        ),
        encoding="utf-8",
    )

    results = BenchmarkRunner(suite_path, output_root=tmp_path / "outputs").run()

    assert len(results["experiments"]) == 1
    record = results["experiments"][0]
    assert record["status"] == "completed"
    assert record["dataset"] == "mvtec_bottle"
    assert record["representation"] == "pixel"
    assert record["normality"] == "diffusion_basic"
    assert record["evidence"] == "noise_residual"
