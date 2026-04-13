"""Tests for multi-seed ablation matrix expansion."""

from pathlib import Path

from adrf.ablation.matrix import AblationMatrix
from adrf.statistics.aggregate import aggregate_grouped_seed_results


def test_multiseed_matrix_expands_each_combo_across_configured_seeds(tmp_path: Path) -> None:
    """A matrix with `seeds` should expand each valid combo into per-seed specs."""

    matrix_path = tmp_path / "multiseed.yaml"
    matrix_path.write_text(
        "\n".join(
            [
                "name: tiny_multiseed",
                "seeds: [0, 1]",
                "defaults:",
                "  protocol:",
                "    alias: one_class",
                "  evaluator:",
                "    alias: basic_ad",
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

    matrix = AblationMatrix.from_yaml(matrix_path)
    specs = matrix.expand()

    assert len(specs) == 2
    assert specs[0]["name"].endswith("__seed0")
    assert specs[1]["name"].endswith("__seed1")
    assert specs[0]["group_name"] == specs[1]["group_name"]
    assert specs[0]["config"]["seed"] == 0
    assert specs[1]["config"]["seed"] == 1


def test_multiseed_grouping_preserves_compatibility_filtered_results() -> None:
    """Grouped aggregation should merge per-seed records under one experiment group."""

    grouped = aggregate_grouped_seed_results(
        [
            {
                "group_name": "combo",
                "experiment_name": "combo__seed0",
                "dataset": "mvtec_bottle",
                "representation": "pixel",
                "normality": "diffusion_basic",
                "evidence": "noise_residual",
                "status": "completed",
                "seed": 0,
                "metrics": {"image_auroc": 0.8},
            },
            {
                "group_name": "combo",
                "experiment_name": "combo__seed1",
                "dataset": "mvtec_bottle",
                "representation": "pixel",
                "normality": "diffusion_basic",
                "evidence": "noise_residual",
                "status": "completed",
                "seed": 1,
                "metrics": {"image_auroc": 0.9},
            },
        ]
    )

    assert grouped[0]["aggregated_metrics"]["image_auroc"]["count"] == 2.0
    assert grouped[0]["seed_runs"][0]["seed"] == 0

