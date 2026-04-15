"""Tests for the ablation runner."""

from pathlib import Path

import yaml

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


def test_ablation_runner_snapshot_preserves_resolved_paper_dataset_config(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "configs" / "dataset"
    dataset_dir.mkdir(parents=True)
    dataset_path = dataset_dir / "mvtec_bottle.yaml"
    dataset_path.write_text(
        "\n".join(
            [
                "name: mvtec_single_class",
                "params:",
                "  root: tests/fixtures/mvtec",
                "  category: bottle",
                "  reference_index: 0",
                "  image_size: [32, 32]",
                "  batch_size: 2",
                "  num_workers: 0",
                "  normalize: false",
            ]
        ),
        encoding="utf-8",
    )
    matrix_path = tmp_path / "paper_matrix.yaml"
    matrix_path.write_text(
        "\n".join(
            [
                "name: paper_snapshot_matrix",
                "continue_on_error: true",
                "datasets:",
                "  - name: mvtec_bottle",
                f"    dataset_config: {dataset_path}",
                "normality: [feature_memory]",
                "evidence: [feature_distance]",
                "representation_map:",
                "  feature_memory: feature",
                "compatibility:",
                "  feature_memory: [feature_distance]",
                "protocol: one_class",
                "evaluation: default",
                "seeds: [0]",
            ]
        ),
        encoding="utf-8",
    )

    results = AblationRunner(matrix_path, output_root=tmp_path / "outputs").run()
    snapshot = yaml.safe_load((Path(results["matrix_dir"]) / "matrix_config_snapshot.yaml").read_text(encoding="utf-8"))

    dataset_path.write_text("name: mutated_dataset\n", encoding="utf-8")

    assert snapshot["expanded_specs"][0]["config"]["datamodule"]["params"]["category"] == "bottle"
