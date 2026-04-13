"""Tests for the ablation runner."""

from pathlib import Path

from adrf.ablation.runner import AblationRunner


def test_ablation_runner_executes_matrix_and_writes_summary(tmp_path: Path) -> None:
    """AblationRunner should run expanded experiments and export a summary."""

    project_root = Path(__file__).resolve().parents[1]
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(
        "\n".join(
            [
                "name: tiny_matrix",
                "continue_on_error: true",
                "defaults:",
                "  protocol:",
                "    name: one_class",
                "    params: {}",
                "  evaluator:",
                "    name: basic_ad",
                "    params: {}",
                "datasets:",
                "  mvtec_bottle:",
                "    name: mvtec_single_class",
                "    params:",
                f"      root: {project_root / 'tests' / 'fixtures' / 'mvtec'}",
                "      category: bottle",
                "      reference_index: 0",
                "      image_size: [32, 32]",
                "      batch_size: 2",
                "      num_workers: 0",
                "      normalize: false",
                "representations:",
                "  pixel:",
                "    name: pixel",
                "    params: {}",
                "normality:",
                "  diffusion_basic:",
                "    name: diffusion_basic",
                "    params:",
                "      input_channels: 3",
                "      hidden_channels: 8",
                "      learning_rate: 0.001",
                "      epochs: 1",
                "      batch_size: 2",
                "      noise_level: 0.2",
                "evidence:",
                "  noise_residual:",
                "    name: noise_residual",
                "    params:",
                "      aggregator: mean",
                "axes:",
                "  datasets: [mvtec_bottle]",
                "  representations: [pixel]",
                "  normality: [diffusion_basic]",
                "  evidence: [noise_residual]",
                "compatibility:",
                "  normality_evidence:",
                "    diffusion_basic: [noise_residual]",
            ]
        ),
        encoding="utf-8",
    )

    results = AblationRunner(matrix_path, output_root=tmp_path / "outputs").run()

    assert results["matrix_name"] == "tiny_matrix"
    assert len(results["experiments"]) == 1
    assert results["experiments"][0]["status"] == "completed"
    suite_dir = Path(results["matrix_dir"])
    assert (suite_dir / "ablation_summary.md").exists()
    assert (suite_dir / "ablation_summary.csv").exists()

