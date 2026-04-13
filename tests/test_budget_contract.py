"""Tests for the audited budget contract."""

from __future__ import annotations

import json
from pathlib import Path

from adrf.ablation.matrix import AblationMatrix
from adrf.runner.experiment_runner import ExperimentRunner


def test_audited_matrix_budget_contract_preserves_shared_overrides_across_seeds(tmp_path: Path) -> None:
    """Matrix expansion should carry the same budget contract into every per-seed spec."""

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

    matrix_path = tmp_path / "audited_matrix.yaml"
    matrix_path.write_text(
        "\n".join(
            [
                "name: paper_baseline_matrix_v3_audited",
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
                "seeds: [0, 1, 2]",
                "overrides:",
                "  trainer:",
                "    max_epochs: 10",
                "  dataloader:",
                "    batch_size: 8",
                "  optimization:",
                "    lr: 0.0001",
                "  dataset:",
                "    image_size: [256, 256]",
                "  diffusion:",
                "    num_steps: 10",
                "  normality:",
                "    backend: legacy",
                "audit:",
                "  export_failure_analysis: true",
            ]
        ),
        encoding="utf-8",
    )

    specs = AblationMatrix.from_yaml(matrix_path).expand()

    assert [spec["seed"] for spec in specs] == [0, 1, 2]
    for spec in specs:
        assert spec["dataset"] == "mvtec_bottle"
        assert spec["evidence"] == "path_cost"
        assert spec["runtime_config"] == "configs/runtime/real.yaml"
        assert spec["overrides"]["trainer"]["max_epochs"] == 10
        assert spec["overrides"]["dataloader"]["batch_size"] == 8
        assert spec["overrides"]["optimization"]["lr"] == 0.0001
        assert spec["overrides"]["dataset"]["image_size"] == [256, 256]
        assert spec["overrides"]["diffusion"]["num_steps"] == 10
        assert spec["overrides"]["normality"]["backend"] == "legacy"


def test_experiment_runner_records_backend_budget_and_audit_paths(tmp_path: Path) -> None:
    """Run metadata should include the effective budget and paths needed for audit export."""

    project_root = Path(__file__).resolve().parents[1]
    config = {
        "name": "audited_budget_demo",
        "runtime_config": str(project_root / "configs" / "runtime" / "smoke.yaml"),
        "overrides": {
            "trainer": {"max_epochs": 2},
            "dataloader": {"batch_size": 2},
            "optimization": {"lr": 0.0005},
            "dataset": {"image_size": [32, 32]},
            "diffusion": {"num_steps": 6},
            "normality": {"backend": "legacy"},
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
                "backend": "legacy",
                "input_channels": 3,
                "hidden_channels": 8,
                "learning_rate": 0.001,
                "epochs": 1,
                "batch_size": 2,
                "noise_level": 0.2,
                "num_steps": 4,
                "step_size": 0.1,
            },
        },
        "evidence": {"name": "path_cost", "params": {"aggregator": "mean"}},
        "evaluator": {"name": "basic_ad", "params": {}},
        "protocol": {"name": "one_class", "params": {}},
    }

    runner = ExperimentRunner(config, output_root=tmp_path / "runs")
    runner.run()

    run_dir = runner.run_dir
    assert run_dir is not None
    run_info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))

    assert run_info["budget"]["max_epochs"] == 2
    assert run_info["budget"]["batch_size"] == 2
    assert run_info["budget"]["lr"] == 0.0005
    assert run_info["budget"]["num_steps"] == 6
    assert run_info["budget"]["backend"] == "legacy"
    assert run_info["artifacts"]["config_snapshot_path"].endswith("config_snapshot.yaml")
    assert run_info["artifacts"]["metrics_path"].endswith("metrics.json")
    assert run_info["artifacts"]["report_path"].endswith("report.md")
