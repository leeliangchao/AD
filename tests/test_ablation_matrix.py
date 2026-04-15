"""Tests for ablation matrix expansion."""

from pathlib import Path

from adrf.ablation.matrix import AblationMatrix


def test_ablation_matrix_expands_valid_grid_and_generates_names(tmp_path: Path) -> None:
    """Matrix configs should expand into named experiment specs with filtering."""

    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(
        "\n".join(
            [
                "name: tiny_matrix",
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
                "      root: ../../tests/fixtures/mvtec",
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
                "  path_cost:",
                "    name: path_cost",
                "    params:",
                "      aggregator: mean",
                "axes:",
                "  datasets: [mvtec_bottle]",
                "  representations: [pixel]",
                "  normality: [diffusion_basic]",
                "  evidence: [noise_residual, path_cost]",
                "compatibility:",
                "  normality_evidence:",
                "    diffusion_basic: [noise_residual]",
            ]
        ),
        encoding="utf-8",
    )

    matrix = AblationMatrix.from_yaml(matrix_path)
    specs = matrix.expand()

    assert len(specs) == 1
    assert specs[0]["name"] == "mvtec_bottle__pixel__diffusion_basic__noise_residual"
    assert specs[0]["config"]["normality"]["name"] == "diffusion_basic"
    assert specs[0]["config"]["evidence"]["name"] == "noise_residual"


def test_ablation_matrix_supports_paper_schema_with_dataset_configs(tmp_path: Path) -> None:
    """Paper-style matrix configs should expand via dataset config files and representation_map."""

    dataset_dir = tmp_path / "configs" / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "mvtec_bottle.yaml").write_text(
        "\n".join(
            [
                "name: mvtec_single_class",
                "params:",
                "  root: ../../tests/fixtures/mvtec",
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
                "name: paper_matrix",
                "datasets:",
                "  - name: mvtec_bottle",
                "    dataset_config: configs/dataset/mvtec_bottle.yaml",
                "normality: [diffusion_basic]",
                "evidence: [noise_residual, path_cost]",
                "representation_map:",
                "  diffusion_basic: pixel",
                "compatibility:",
                "  diffusion_basic: [noise_residual]",
                "protocol: one_class",
                "evaluation: default",
                "seeds: [0]",
            ]
        ),
        encoding="utf-8",
    )

    matrix = AblationMatrix.from_yaml(matrix_path)
    specs = matrix.expand()

    assert len(specs) == 1
    assert specs[0]["name"] == "mvtec_bottle__pixel__diffusion_basic__noise_residual__seed0"
    assert specs[0]["config"]["datamodule"]["name"] == "mvtec_single_class"
    assert specs[0]["config"]["protocol"]["name"] == "one_class"
    assert specs[0]["config"]["evaluator"]["name"] == "basic_ad"


def test_ablation_matrix_distinguishes_row_level_override_variants_in_identity(tmp_path: Path) -> None:
    matrix_path = tmp_path / "rows.yaml"
    matrix_path.write_text(
        "\n".join(
            [
                "name: override_identity_matrix",
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
                "      root: ../../tests/fixtures/mvtec",
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
                "  diffusion_inversion_basic:",
                "    name: diffusion_inversion_basic",
                "    params:",
                "      input_channels: 3",
                "      hidden_channels: 8",
                "      learning_rate: 0.001",
                "      epochs: 1",
                "      batch_size: 2",
                "      noise_level: 0.2",
                "      num_steps: 4",
                "      step_size: 0.1",
                "evidence:",
                "  path_cost:",
                "    name: path_cost",
                "    params:",
                "      aggregator: mean",
                "rows:",
                "  - dataset: mvtec_bottle",
                "    representation: pixel",
                "    normality: diffusion_inversion_basic",
                "    evidence: path_cost",
                "    overrides:",
                "      normality:",
                "        num_steps: 4",
                "  - dataset: mvtec_bottle",
                "    representation: pixel",
                "    normality: diffusion_inversion_basic",
                "    evidence: path_cost",
                "    overrides:",
                "      normality:",
                "        num_steps: 10",
            ]
        ),
        encoding="utf-8",
    )

    matrix = AblationMatrix.from_yaml(matrix_path)
    specs = matrix.expand()

    assert len(specs) == 2
    assert len({spec["name"] for spec in specs}) == 2
    assert len({spec["group_name"] for spec in specs}) == 2
