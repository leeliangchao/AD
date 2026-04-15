from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch import nn

from adrf.data.datamodule import MVTecDataModule
from adrf.normality.base import BaseNormalityModel
from adrf.protocol.one_class import OneClassProtocol
from adrf.protocol.results import TrainSummary
from adrf.representation.base import BaseRepresentation
from adrf.representation.contracts import RepresentationBatch, RepresentationProvenance


def _write_rgb_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def _build_fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "mvtec"
    _write_rgb_image(root / "bottle" / "train" / "good" / "000.png", (0, 0, 0))
    _write_rgb_image(root / "bottle" / "train" / "good" / "001.png", (16, 16, 16))
    _write_rgb_image(root / "bottle" / "test" / "good" / "002.png", (0, 0, 0))
    return root


class _TrainableToyRepresentation(BaseRepresentation):
    space = "feature"
    trainable = True

    def __init__(self) -> None:
        super().__init__(input_image_size=(32, 32), input_normalize=False)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.encode_batch_calls = 0

    def encode_batch(self, samples: Sequence) -> RepresentationBatch:
        self.encode_batch_calls += 1
        return super().encode_batch(samples)

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
            config_fingerprint="toy-trainable-feature",
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
        raise AssertionError("joint mode should not call fit().")

    def infer(self, sample, representation):
        raise AssertionError("test does not exercise inference.")


class _BatchNormEvalRepresentation(BaseRepresentation):
    space = "feature"
    trainable = True

    def __init__(self) -> None:
        super().__init__(input_image_size=(32, 32), input_normalize=False)
        self.batch_norm = nn.BatchNorm2d(1)

    def _encode_tensor_batch(self, batch: torch.Tensor) -> torch.Tensor:
        return self.batch_norm(batch[:, :1, :, :])

    def describe(self) -> RepresentationProvenance:
        return RepresentationProvenance(
            representation_name="feature",
            backbone_name="toy-bn",
            weights_source=None,
            feature_layer="batch_norm",
            pooling=None,
            trainable=True,
            frozen_submodules=(),
            input_image_size=self.input_image_size,
            input_normalize=self.input_normalize,
            normalize_mean=None,
            normalize_std=None,
            code_version="tests",
            config_fingerprint="toy-batchnorm-eval-feature",
        )


class _EvalAwareNormality(nn.Module, BaseNormalityModel):
    fit_mode = "joint"
    accepted_spaces = frozenset({"feature"})
    accepted_tensor_ranks = frozenset({3})
    requires_detached_representation = False

    def __init__(self) -> None:
        super().__init__()
        self.infer_training_states: list[bool] = []

    def configure_joint_training(self, representation_model: nn.Module) -> None:
        del representation_model

    def fit_batch(self, representations: RepresentationBatch, samples) -> dict[str, float]:
        del representations, samples
        return {"loss": 0.0}

    def fit(self, representations, samples=None) -> None:
        del representations, samples
        raise AssertionError("joint mode should not call fit().")

    def infer(self, sample, representation):
        del sample, representation
        self.infer_training_states.append(self.training)
        return object()


class _DistributedSummaryJointNormality(nn.Module, BaseNormalityModel):
    fit_mode = "joint"
    accepted_spaces = frozenset({"feature"})
    accepted_tensor_ranks = frozenset({3})
    requires_detached_representation = False

    def configure_joint_training(self, representation_model: nn.Module) -> None:
        del representation_model

    def fit_batch(self, representations: RepresentationBatch, samples) -> dict[str, float]:
        del representations, samples
        return {"loss": 1.0}

    def fit(self, representations, samples=None) -> None:
        del representations, samples
        raise AssertionError("joint mode should not call fit().")

    def infer(self, sample, representation):
        del sample, representation
        raise AssertionError("test does not exercise inference.")


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


def test_one_class_protocol_joint_fit_updates_trainable_representation_parameter(tmp_path: Path) -> None:
    torch.manual_seed(0)
    root = _build_fixture_root(tmp_path)
    representation = _TrainableToyRepresentation()
    normality = _JointToyNormality()
    runner = SimpleNamespace(
        datamodule=MVTecDataModule(
            root=root,
            category="bottle",
            image_size=(32, 32),
            batch_size=2,
            num_workers=0,
            normalize=False,
        ),
        representation=representation,
        normality=normality,
    )
    protocol = OneClassProtocol()
    initial_scale = representation.scale.detach().clone()

    train_summary = protocol.train_epoch(runner)

    assert train_summary["num_train_samples"] == 2
    assert train_summary["num_train_batches"] == 1
    assert representation.encode_batch_calls == 1
    assert not torch.allclose(representation.scale.detach(), initial_scale)


def test_one_class_protocol_evaluate_temporarily_switches_trainable_modules_to_eval(tmp_path: Path) -> None:
    root = _build_fixture_root(tmp_path)
    representation = _BatchNormEvalRepresentation()
    normality = _EvalAwareNormality()
    runner = SimpleNamespace(
        datamodule=MVTecDataModule(
            root=root,
            category="bottle",
            image_size=(32, 32),
            batch_size=1,
            num_workers=0,
            normalize=False,
        ),
        representation=representation,
        normality=normality,
        evidence=_NoopEvidence(),
        evaluator=_CountingEvaluator(),
    )
    protocol = OneClassProtocol()
    before_batches = int(representation.batch_norm.num_batches_tracked.item())

    metrics = protocol.evaluate(runner)

    assert metrics == {"image_auroc": 1.0}
    assert representation.training is True
    assert representation.batch_norm.training is True
    assert normality.training is True
    assert normality.infer_training_states == [False]
    assert int(representation.batch_norm.num_batches_tracked.item()) == before_batches

class _DistributedSummaryJointNormality(nn.Module, BaseNormalityModel):
    fit_mode = "joint"
    accepted_spaces = frozenset({"feature"})
    accepted_tensor_ranks = frozenset({3})
    requires_detached_representation = False

    def configure_joint_training(self, representation_model: nn.Module) -> None:
        del representation_model

    def fit_batch(self, representations: RepresentationBatch, samples) -> dict[str, float]:
        del representations, samples
        return {"loss": 1.0}

    def fit(self, representations, samples=None) -> None:
        del representations, samples
        raise AssertionError("joint mode should not call fit().")

    def infer(self, sample, representation):
        del sample, representation
        raise AssertionError("test does not exercise inference.")
def test_one_class_protocol_joint_fit_aggregates_distributed_train_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _build_fixture_root(tmp_path)
    runner = SimpleNamespace(
        datamodule=MVTecDataModule(
            root=root,
            category="bottle",
            image_size=(32, 32),
            batch_size=2,
            num_workers=0,
            normalize=False,
        ),
        representation=_TrainableToyRepresentation(),
        normality=_DistributedSummaryJointNormality(),
        distributed_context=SimpleNamespace(enabled=True, world_size=2),
        distributed_training_enabled=True,
    )
    protocol = OneClassProtocol()

    monkeypatch.setattr(
        "adrf.protocol.runtime_support.all_gather_objects",
        lambda payload, context: [
            payload,
            TrainSummary(
                num_train_batches=2,
                num_train_samples=3,
                metrics={"loss": 4.0},
                metric_weights={"loss": 2.0},
            ),
        ],
    )

    train_summary = protocol.train_epoch(runner)

    assert train_summary == {
        "num_train_batches": 3,
        "num_train_samples": 5,
        "loss": 3.0,
    }
