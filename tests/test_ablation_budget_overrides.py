"""Tests for budget-aligned ablation override propagation."""

from __future__ import annotations

import json
from pathlib import Path

from adrf.ablation.matrix import AblationMatrix
from adrf.runner.experiment_runner import ExperimentRunner


def test_paper_matrix_expands_matrix_level_budget_overrides(tmp_path: Path) -> None:
    """Paper matrices should attach shared budget overrides to every expanded spec."""

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
                "name: budgeted_paper_matrix",
                "runtime_config: configs/runtime/real.yaml",
                "datasets:",
                "  - name: mvtec_bottle",
                "    dataset_config: configs/dataset/mvtec_bottle.yaml",
                "normality: [diffusion_inversion_basic]",
                "evidence: [path_cost]",
                "representation_map:",
                "  diffusion_inversion_basic: pixel",
                "compatibility:",
                "  diffusion_inversion_basic: [path_cost]",
                "protocol: one_class",
                "evaluation: default",
                "seeds: [0, 1]",
                "overrides:",
                "  trainer:",
                "    max_epochs: 5",
                "  dataloader:",
                "    batch_size: 4",
                "  optimization:",
                "    lr: 0.0003",
                "  dataset:",
                "    image_size: [16, 16]",
                "  diffusion:",
                "    num_steps: 7",
                "  runtime_config: configs/runtime/smoke.yaml",
            ]
        ),
        encoding="utf-8",
    )

    matrix = AblationMatrix.from_yaml(matrix_path)
    specs = matrix.expand()

    assert len(specs) == 2
    for spec in specs:
        assert spec["dataset"] == "mvtec_bottle"
        assert spec["normality"] == "diffusion_inversion_basic"
        assert spec["evidence"] == "path_cost"
        assert spec["runtime_config"] == "configs/runtime/smoke.yaml"
        assert spec["overrides"]["trainer"]["max_epochs"] == 5
        assert spec["overrides"]["dataloader"]["batch_size"] == 4
        assert spec["overrides"]["optimization"]["lr"] == 0.0003
        assert spec["overrides"]["dataset"]["image_size"] == [16, 16]
        assert spec["overrides"]["diffusion"]["num_steps"] == 7
        assert spec["config"]["seed"] in {0, 1}


def test_experiment_runner_applies_and_records_budget_overrides(tmp_path: Path) -> None:
    """ExperimentRunner should apply normalized budget overrides and persist them."""

    project_root = Path(__file__).resolve().parents[1]
    runtime_config = str(project_root / "configs" / "runtime" / "smoke.yaml")
    config = {
        "name": "budget_override_demo",
        "runtime_config": str(project_root / "configs" / "runtime" / "real.yaml"),
        "overrides": {
            "trainer": {"max_epochs": 2},
            "dataloader": {"batch_size": 1},
            "optimization": {"lr": 0.005},
            "dataset": {"image_size": [16, 16]},
            "normality": {"num_steps": 5},
            "runtime_config": runtime_config,
        },
        "datamodule": {
            "name": "mvtec_single_class",
            "params": {
                "root": str(project_root / "tests" / "fixtures" / "mvtec"),
                "category": "bottle",
                "reference_index": 0,
                "image_size": [32, 32],
                "batch_size": 2,
                "num_workers": 0,
                "normalize": False,
            },
        },
        "representation": {"name": "pixel", "params": {}},
        "normality": {
            "name": "diffusion_inversion_basic",
            "params": {
                "input_channels": 3,
                "hidden_channels": 8,
                "learning_rate": 0.001,
                "epochs": 1,
                "batch_size": 2,
                "noise_level": 0.2,
                "num_steps": 3,
                "step_size": 0.1,
            },
        },
        "evidence": {"name": "path_cost", "params": {"aggregator": "mean"}},
        "evaluator": {"name": "basic_ad", "params": {}},
        "protocol": {"name": "one_class", "params": {}},
    }

    runner = ExperimentRunner(config, output_root=tmp_path / "runs")
    runner.run()

    assert runner.datamodule is not None
    assert runner.normality is not None
    assert runner.datamodule.batch_size == 1
    assert runner.datamodule.image_size == [16, 16]
    assert runner.normality.batch_size == 1
    assert runner.normality.epochs == 2
    assert runner.normality.learning_rate == 0.005
    assert runner.normality.num_steps == 5

    run_info = json.loads((runner.run_dir / "run_info.json").read_text(encoding="utf-8"))
    assert run_info["budget"]["max_epochs"] == 2
    assert run_info["budget"]["batch_size"] == 1
    assert run_info["budget"]["lr"] == 0.005
    assert run_info["budget"]["num_steps"] == 5
    assert run_info["budget"]["image_size"] == [16, 16]
    assert run_info["budget"]["runtime_config"] == runtime_config


def test_experiment_runner_skips_unsupported_budget_params() -> None:
    """Budget overrides should not inject unsupported params into non-trainable normality families."""

    project_root = Path(__file__).resolve().parents[1]
    config = {
        "name": "feature_memory_budget_guard",
        "runtime_config": str(project_root / "configs" / "runtime" / "smoke.yaml"),
        "overrides": {
            "trainer": {"max_epochs": 2},
            "dataloader": {"batch_size": 4},
            "optimization": {"lr": 0.005},
            "normality": {"num_steps": 5},
        },
        "datamodule": {
            "name": "mvtec_single_class",
            "params": {
                "root": str(project_root / "tests" / "fixtures" / "mvtec"),
                "category": "bottle",
                "reference_index": 0,
                "image_size": [32, 32],
                "batch_size": 2,
                "num_workers": 0,
                "normalize": False,
            },
        },
        "representation": {"name": "feature", "params": {"pretrained": False, "freeze": True}},
        "normality": {"name": "feature_memory", "params": {}},
        "evidence": {"name": "feature_distance", "params": {"aggregator": "max"}},
        "evaluator": {"name": "basic_ad", "params": {}},
        "protocol": {"name": "one_class", "params": {}},
    }

    runner = ExperimentRunner(config)
    runner.setup()

    assert runner.normality is not None
    assert runner.config["normality"]["params"] == {}
    assert runner.config["datamodule"]["params"]["batch_size"] == 4
    assert runner.budget_info["batch_size"] == 4
