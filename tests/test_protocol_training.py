from types import SimpleNamespace

import torch
from torch import nn

from adrf.core.sample import Sample
from adrf.protocol.context import ProtocolContext
from adrf.protocol.results import TrainSummary
from adrf.protocol.training import JointTrainingStrategy, OfflineTrainingStrategy, resolve_training_strategy
from adrf.representation.contracts import RepresentationBatch, RepresentationProvenance


def _provenance() -> RepresentationProvenance:
    return RepresentationProvenance(
        representation_name="feature",
        backbone_name="toy",
        weights_source=None,
        feature_layer="unit",
        pooling=None,
        trainable=False,
        frozen_submodules=(),
        input_image_size=(8, 8),
        input_normalize=False,
        normalize_mean=None,
        normalize_std=None,
        code_version="tests",
        config_fingerprint="protocol-training",
    )


def _make_samples(prefix: str, count: int) -> list[Sample]:
    return [
        Sample(image=torch.zeros(3, 8, 8), sample_id=f"{prefix}-{index}")
        for index in range(count)
    ]


def _make_representation_batch(samples) -> RepresentationBatch:
    return RepresentationBatch(
        tensor=torch.zeros(len(samples), 1, 8, 8),
        space="feature",
        spatial_shape=(8, 8),
        feature_dim=1,
        batch_size=len(samples),
        sample_ids=tuple(sample.sample_id for sample in samples),
        requires_grad=False,
        device="cpu",
        dtype="torch.float32",
        provenance=_provenance(),
    )


class _Representation:
    def __init__(self) -> None:
        self.calls = 0
        self.representation = self

    def encode_batch(self, samples):
        self.calls += 1
        return _make_representation_batch(samples)


class _OfflineNormality:
    fit_mode = "offline"

    def __init__(self) -> None:
        self.fit_calls = []

    def fit(self, representations, samples=None) -> None:
        self.fit_calls.append((list(representations), list(samples or [])))


class _JointNormality(nn.Module):
    fit_mode = "joint"

    def __init__(self) -> None:
        super().__init__()
        self.configured_with = None
        self.fit_batch_calls = 0

    def configure_joint_training(self, representation_model) -> None:
        self.configured_with = representation_model

    def fit_batch(self, representations, samples) -> dict[str, float]:
        del representations, samples
        self.fit_batch_calls += 1
        return {"loss": float(self.fit_batch_calls)}


class _SequenceJointNormality(nn.Module):
    fit_mode = "joint"

    def __init__(self, batch_metrics: list[dict[str, float]]) -> None:
        super().__init__()
        self.batch_metrics = batch_metrics
        self.configured_with = None
        self.fit_batch_calls = 0

    def configure_joint_training(self, representation_model) -> None:
        self.configured_with = representation_model

    def fit_batch(self, representations, samples) -> dict[str, float]:
        del representations, samples
        metrics = self.batch_metrics[self.fit_batch_calls]
        self.fit_batch_calls += 1
        return metrics


def _make_context(normality, *, distributed_context=None, distributed_training_enabled=False) -> ProtocolContext:
    representation = _Representation()
    samples = _make_samples("train", 2)
    return ProtocolContext(
        train_loader=[samples],
        test_loader=[],
        representation=representation,
        normality=normality,
        evidence=object(),
        evaluator=object(),
        distributed_context=distributed_context,
        distributed_training_enabled=distributed_training_enabled,
    )


def test_resolve_training_strategy_uses_legacy_fit_mode_contract() -> None:
    assert isinstance(resolve_training_strategy(_make_context(_OfflineNormality())), OfflineTrainingStrategy)
    assert isinstance(resolve_training_strategy(_make_context(_JointNormality())), JointTrainingStrategy)


def test_offline_training_strategy_fits_gathered_representations_and_merges_summary(monkeypatch) -> None:
    normality = _OfflineNormality()
    context = _make_context(
        normality,
        distributed_context=SimpleNamespace(enabled=True, world_size=2),
        distributed_training_enabled=False,
    )
    remote_samples = _make_samples("remote", 3)

    monkeypatch.setattr(
        "adrf.protocol.training.all_gather_objects",
        lambda payload, distributed_context: [
            payload,
            {
                "samples": remote_samples,
                "representations": _make_representation_batch(remote_samples).unbind(),
            },
        ],
    )
    monkeypatch.setattr(
        "adrf.protocol.runtime_support.all_gather_objects",
        lambda payload, distributed_context: [
            payload,
            TrainSummary(num_train_batches=2, num_train_samples=3),
        ],
    )

    summary = OfflineTrainingStrategy().train(context)

    assert summary == TrainSummary(num_train_batches=3, num_train_samples=5, metrics={})
    assert len(normality.fit_calls) == 1
    assert len(normality.fit_calls[0][0]) == 5
    assert len(normality.fit_calls[0][1]) == 5


def test_joint_training_strategy_tracks_per_metric_batch_counts() -> None:
    normality = _SequenceJointNormality(
        [
            {"loss": 2.0, "aux": 4.0},
            {"loss": 6.0},
        ]
    )
    representation = _Representation()
    context = ProtocolContext(
        train_loader=[
            _make_samples("train", 2),
            _make_samples("train-extra", 1),
        ],
        test_loader=[],
        representation=representation,
        normality=normality,
        evidence=object(),
        evaluator=object(),
        distributed_context=None,
        distributed_training_enabled=False,
    )

    summary = JointTrainingStrategy().train(context)

    assert normality.configured_with is context.representation.representation
    assert normality.fit_batch_calls == 2
    assert summary.metric_weights == {"loss": 2.0, "aux": 1.0}
    assert summary == TrainSummary(
        num_train_batches=2,
        num_train_samples=3,
        metrics={"loss": 4.0, "aux": 4.0},
        metric_weights={"loss": 2.0, "aux": 1.0},
    )
