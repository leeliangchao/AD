"""Tests for budget-aligned ablation override propagation."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from adrf.ablation.matrix import AblationMatrix
from adrf.normality.base import BaseNormalityModel
from adrf.representation.base import BaseRepresentation
from adrf.representation.contracts import RepresentationBatch, RepresentationProvenance
from adrf.runner.experiment_runner import ExperimentRunner, build_default_registry


class _TrainableToyRepresentation(BaseRepresentation):
    space = "feature"
    trainable = True

    def __init__(self) -> None:
        super().__init__(input_image_size=(32, 32), input_normalize=False)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def _encode_tensor_batch(self, batch: torch.Tensor) -> torch.Tensor:
        return batch[:, :1, :, :] * self.scale

    def describe(self) -> RepresentationProvenance:
        return RepresentationProvenance(
            representation_name="feature",
            backbone_name="toy",
            weights_source=None,
            feature_layer="scale",
            pooling=None,
            trainable=True,
            frozen_submodules=(),
            input_image_size=self.input_image_size,
            input_normalize=self.input_normalize,
            normalize_mean=None,
            normalize_std=None,
            code_version="tests",
            config_fingerprint="budget-overrides-toy-feature",
        )


class _JointToyNormality(nn.Module, BaseNormalityModel):
    fit_mode = "joint"
    accepted_spaces = frozenset({"feature"})
    accepted_tensor_ranks = frozenset({3})
    requires_detached_representation = False

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.optimizer: torch.optim.Optimizer | None = None

    def configure_joint_training(self, representation_model: nn.Module) -> None:
        self.optimizer = torch.optim.SGD(
            list(self.parameters())
            + [parameter for parameter in representation_model.parameters() if parameter.requires_grad],
            lr=1e-2,
        )

    def fit_batch(self, representations: RepresentationBatch, samples) -> dict[str, float]:
        del samples
        self.validate_representation_batch(representations)
        assert self.optimizer is not None
        loss = (representations.tensor.mean() * self.scale).square()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.detach().item())}

    def fit(self, representations, samples=None) -> None:
        del representations, samples
        raise AssertionError("joint mode should not call fit().")

    def infer(self, sample, representation):
        del sample, representation
        return object()


class _NoopEvidence:
    def predict(self, sample, artifacts):
        del sample, artifacts
        return {"image_score": 0.0}


class _CountingEvaluator:
    def __init__(self) -> None:
        self.updates = 0

    def reset(self) -> None:
        self.updates = 0

    def update(self, prediction, sample) -> None:
        del prediction, sample
        self.updates += 1

    def compute(self) -> dict[str, float]:
        return {"image_auroc": float(self.updates)}

    def state_dict(self):
        return {"updates": self.updates}

    def merge_states(self, states):
        return states[0]

    def load_state_dict(self, state):
        self.updates = int(state["updates"])


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


def test_experiment_runner_persists_representation_provenance_and_checkpoint(tmp_path: Path) -> None:
    """Trainable representations should persist provenance metadata and their own checkpoint."""

    project_root = Path(__file__).resolve().parents[1]
    config = {
        "name": "trainable_representation_persistence",
        "runtime_config": str(project_root / "configs" / "runtime" / "smoke.yaml"),
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
        "representation": {"name": "toy_trainable_feature", "params": {}},
        "normality": {"name": "toy_joint_normality", "params": {}},
        "evidence": {"name": "noop_evidence", "params": {}},
        "evaluator": {"name": "counting_evaluator", "params": {}},
        "protocol": {"name": "one_class", "params": {}},
    }

    registry = build_default_registry()
    registry.register("representation", "toy_trainable_feature", _TrainableToyRepresentation)
    registry.register("normality", "toy_joint_normality", _JointToyNormality)
    registry.register("evidence", "noop_evidence", _NoopEvidence)
    registry.register("evaluator", "counting_evaluator", _CountingEvaluator)

    runner = ExperimentRunner(config, registry=registry, output_root=tmp_path / "runs")
    runner.run()

    assert runner.run_dir is not None
    run_info = json.loads((runner.run_dir / "run_info.json").read_text(encoding="utf-8"))
    assert run_info["representation"]["provenance"]["representation_name"] == "feature"
    assert run_info["representation"]["provenance"]["backbone_name"] == "toy"
    assert run_info["representation"]["provenance"]["trainable"] is True
    assert run_info["representation"]["provenance"]["input_image_size"] == [32, 32]
    assert (runner.run_dir / "checkpoints" / "normality.pt").exists()
    assert (runner.run_dir / "checkpoints" / "representation.pt").exists()


def test_paper_matrix_feature_representation_follows_effective_dataset_shape_and_normalize(tmp_path: Path) -> None:
    """Paper-schema feature baselines should inherit effective dataset sizing and normalize settings."""

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

    matrix_path = tmp_path / "paper_feature_matrix.yaml"
    matrix_path.write_text(
        "\n".join(
            [
                "name: paper_feature_matrix",
                "runtime_config: configs/runtime/smoke.yaml",
                "datasets:",
                "  - name: mvtec_bottle",
                "    dataset_config: configs/dataset/mvtec_bottle.yaml",
                "normality: [feature_memory]",
                "evidence: [feature_distance]",
                "representation_map:",
                "  feature_memory: feature",
                "compatibility:",
                "  feature_memory: [feature_distance]",
                "protocol: one_class",
                "evaluation: default",
                "overrides:",
                "  dataset:",
                "    image_size: [48, 40]",
                "    normalize: true",
            ]
        ),
        encoding="utf-8",
    )

    [spec] = AblationMatrix.from_yaml(matrix_path).expand()
    params = spec["config"]["representation"]["params"]

    assert params["weights"] == "imagenet1k_v1"
    assert params["trainable"] is False
    assert tuple(params["input_image_size"]) == (48, 40)
    assert params["input_normalize"] is True
    assert "pretrained" not in params
    assert "freeze" not in params
