"""Tests for the official baseline contract."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from adrf.ablation.matrix import AblationMatrix
from adrf.ablation.runner import AblationRunner
from adrf.runner.experiment_runner import ExperimentRunner, build_default_registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_MATRIX_PATH = PROJECT_ROOT / "configs" / "ablation" / "paper_baseline_matrix_official_v1.yaml"
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "mvtec"
EXPECTED_CORE_COMBOS = {
    ("feature_memory", "feature_distance"),
    ("autoencoder", "reconstruction_residual"),
    ("diffusion_basic", "noise_residual"),
    ("diffusion_inversion_basic", "path_cost"),
    ("diffusion_inversion_basic", "direction_mismatch"),
    ("reference_basic", "conditional_violation"),
    ("reference_diffusion_basic", "noise_residual"),
}


def test_experiment_runner_uses_protocol_run_as_official_contract(tmp_path: Path) -> None:
    """ExperimentRunner.run should delegate the lifecycle to protocol.run(runner)."""

    class SpyProtocol:
        def __init__(self) -> None:
            self.run_calls = 0

        def train_epoch(self, runner: object) -> dict[str, object]:
            raise AssertionError("ExperimentRunner.run should not call train_epoch directly.")

        def evaluate(self, runner: object) -> dict[str, float]:
            raise AssertionError("ExperimentRunner.run should not call evaluate directly.")

        def run(self, runner: object) -> dict[str, object]:
            self.run_calls += 1
            return {
                "train": {"num_train_samples": 0},
                "evaluation": {
                    "image_auroc": 0.5,
                    "pixel_auroc": 0.5,
                    "pixel_aupr": 0.5,
                },
            }

    registry = build_default_registry()
    registry.register("protocol", "spy_protocol", SpyProtocol)
    config = {
        "name": "protocol_contract_demo",
        "datamodule": {
            "name": "mvtec_single_class",
            "params": {
                "root": str(FIXTURE_ROOT),
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
            "name": "autoencoder",
            "params": {
                "input_channels": 3,
                "hidden_channels": 4,
                "latent_channels": 8,
                "epochs": 1,
                "batch_size": 2,
            },
        },
        "evidence": {"name": "reconstruction_residual", "params": {"aggregator": "mean"}},
        "evaluator": {"name": "basic_ad", "params": {}},
        "protocol": {"name": "spy_protocol", "params": {}},
    }

    runner = ExperimentRunner(config, registry=registry, output_root=tmp_path / "runs")
    results = runner.run()

    assert results["train"]["num_train_samples"] == 0
    assert results["evaluation"]["image_auroc"] == 0.5
    assert runner.protocol.run_calls == 1


def test_official_matrix_expands_fixed_contract() -> None:
    """The official matrix should expand the fixed dataset/seed/core-combo contract."""

    matrix = AblationMatrix.from_yaml(OFFICIAL_MATRIX_PATH)
    specs = matrix.expand()

    assert matrix.name == "paper_baseline_matrix_official_v1"
    assert sorted({entry["name"] for entry in matrix.config["datasets"]}) == [
        "mvtec_bottle",
        "mvtec_capsule",
        "mvtec_grid",
    ]
    assert matrix.config["seeds"] == [0, 1, 2]
    assert matrix.config["runtime_config"] == "configs/runtime/real.yaml"
    assert len(specs) == 63
    assert {spec["seed"] for spec in specs} == {0, 1, 2}
    assert {(spec["normality"], spec["evidence"]) for spec in specs} == EXPECTED_CORE_COMBOS

    for spec in specs:
        assert spec["runtime_config"] == "configs/runtime/real.yaml"
        assert spec["config"]["protocol"]["name"] == "one_class"
        assert spec["config"]["evaluator"]["name"] == "basic_ad"
        assert spec["overrides"]["trainer"]["max_epochs"] == 10
        assert spec["overrides"]["dataloader"]["batch_size"] == 32
        assert spec["overrides"]["optimization"]["lr"] == 1.0e-4
        assert spec["overrides"]["dataset"]["image_size"] == [256, 256]
        assert spec["overrides"]["diffusion"]["num_steps"] == 10
        assert spec["overrides"]["normality"]["backend"] == "legacy"


def test_official_contract_metrics_are_emitted_by_matrix_runs(tmp_path: Path) -> None:
    """A reduced official-contract matrix should emit the fixed comparison metrics."""

    dataset_dir = tmp_path / "configs" / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "mvtec_bottle.yaml").write_text(
        "\n".join(
            [
                "name: mvtec_single_class",
                "params:",
                f"  root: {FIXTURE_ROOT.as_posix()}",
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

    official_config = deepcopy(AblationMatrix.from_yaml(OFFICIAL_MATRIX_PATH).config)
    official_config["name"] = "paper_baseline_matrix_official_v1_smoke"
    official_config["datasets"] = [
        {
            "name": "mvtec_bottle",
            "dataset_config": str((dataset_dir / "mvtec_bottle.yaml").resolve()),
        }
    ]
    official_config["normality"] = ["feature_memory"]
    official_config["evidence"] = ["feature_distance"]
    official_config["compatibility"] = {"feature_memory": ["feature_distance"]}
    official_config["seeds"] = [0]

    matrix_path = tmp_path / "official_matrix_smoke.yaml"
    matrix_path.write_text(yaml.safe_dump(official_config, sort_keys=False), encoding="utf-8")

    results = AblationRunner(matrix_path, output_root=tmp_path / "outputs").run()

    record = results["experiments"][0]
    assert record["status"] == "completed"
    assert {"image_auroc", "pixel_auroc", "pixel_aupr", "train_time", "total_time"} <= set(record["metrics"])

