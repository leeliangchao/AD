"""Tests for benchmark suite loading."""

from pathlib import Path

from adrf.benchmark.suite import BenchmarkSuite


def test_benchmark_suite_loads_relative_experiment_paths(tmp_path: Path) -> None:
    """BenchmarkSuite should resolve experiment config paths relative to the suite file."""

    suite_path = tmp_path / "suite.yaml"
    experiments_dir = tmp_path / "configs"
    experiments_dir.mkdir()
    (experiments_dir / "a.yaml").write_text("representation:\n  name: pixel\n", encoding="utf-8")
    (experiments_dir / "b.yaml").write_text("representation:\n  name: pixel\n", encoding="utf-8")
    suite_path.write_text(
        "\n".join(
            [
                "name: demo_suite",
                "continue_on_error: true",
                "experiments:",
                "  - configs/a.yaml",
                "  - configs/b.yaml",
            ]
        ),
        encoding="utf-8",
    )

    suite = BenchmarkSuite.from_yaml(suite_path)

    assert suite.name == "demo_suite"
    assert suite.continue_on_error is True
    assert len(suite.experiments) == 2
    assert all(path.exists() for path in suite.experiments)

