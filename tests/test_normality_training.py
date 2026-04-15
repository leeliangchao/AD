from __future__ import annotations

from types import SimpleNamespace

import pytest
from torch import nn
import torch

from adrf.normality.training import (
    JointNormalityTrainingAdapter,
    OfflineNormalityTrainingAdapter,
    resolve_normality_training_adapter,
    validate_normality_representation_contract,
)
from adrf.representation.contracts import RepresentationOutput, RepresentationProvenance


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
        self.fit_batch_calls = []

    def configure_joint_training(self, representation_model) -> None:
        self.configured_with = representation_model

    def fit_batch(self, representations, samples) -> dict[str, float]:
        self.fit_batch_calls.append((representations, samples))
        return {"loss": 1.0}


class _UnsupportedModeNormality:
    fit_mode = "weird"


class _IncompleteOfflineNormality:
    fit_mode = "offline"


class _IncompleteJointNormality:
    fit_mode = "joint"


class _FeatureOfflineNormality(_OfflineNormality):
    accepted_spaces = frozenset({"feature"})
    accepted_tensor_ranks = frozenset({3})
    requires_detached_representation = True


def _representation_output(
    *,
    space: str = "feature",
    tensor: torch.Tensor | None = None,
    requires_grad: bool | None = None,
) -> RepresentationOutput:
    resolved_tensor = tensor if tensor is not None else torch.ones(4, 2, 2)
    resolved_requires_grad = bool(resolved_tensor.requires_grad if requires_grad is None else requires_grad)
    provenance = RepresentationProvenance(
        representation_name=space,
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
        config_fingerprint="normality-training",
    )
    return RepresentationOutput(
        tensor=resolved_tensor,
        space=space,
        spatial_shape=tuple(resolved_tensor.shape[-2:]) if resolved_tensor.ndim == 3 else None,
        feature_dim=int(resolved_tensor.shape[0]),
        sample_id="sample-001",
        requires_grad=resolved_requires_grad,
        device=str(resolved_tensor.device),
        dtype=str(resolved_tensor.dtype),
        provenance=provenance,
    )


def test_resolve_normality_training_adapter_supports_offline_and_joint_modes() -> None:
    assert isinstance(resolve_normality_training_adapter(_OfflineNormality()), OfflineNormalityTrainingAdapter)
    assert isinstance(resolve_normality_training_adapter(_JointNormality()), JointNormalityTrainingAdapter)


def test_offline_normality_training_adapter_delegates_fit() -> None:
    normality = _OfflineNormality()
    adapter = resolve_normality_training_adapter(normality)

    adapter.fit(["representation"], ["sample"])

    assert normality.fit_calls == [(["representation"], ["sample"])]


def test_joint_normality_training_adapter_delegates_configure_and_fit_batch() -> None:
    normality = _JointNormality()
    adapter = resolve_normality_training_adapter(normality)

    adapter.configure_joint_training("representation-model")
    metrics = adapter.fit_batch("batch-representations", ["sample"])

    assert normality.configured_with == "representation-model"
    assert normality.fit_batch_calls == [("batch-representations", ["sample"])]
    assert metrics == {"loss": 1.0}


def test_resolve_normality_training_adapter_rejects_unsupported_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported normality fit mode"):
        resolve_normality_training_adapter(_UnsupportedModeNormality())


def test_resolve_normality_training_adapter_rejects_incomplete_offline_contract() -> None:
    with pytest.raises(RuntimeError, match="must implement offline training fit"):
        resolve_normality_training_adapter(_IncompleteOfflineNormality())


def test_resolve_normality_training_adapter_rejects_incomplete_joint_contract() -> None:
    with pytest.raises(RuntimeError, match="must implement joint training"):
        resolve_normality_training_adapter(_IncompleteJointNormality())


def test_validate_normality_representation_contract_accepts_matching_offline_contract() -> None:
    representation = SimpleNamespace(space="feature")
    probe_output = _representation_output()

    validate_normality_representation_contract(_FeatureOfflineNormality(), representation, probe_output)


def test_validate_normality_representation_contract_rejects_space_mismatch() -> None:
    representation = SimpleNamespace(space="pixel")
    probe_output = _representation_output()

    with pytest.raises(ValueError, match="requires representation space"):
        validate_normality_representation_contract(_FeatureOfflineNormality(), representation, probe_output)


def test_validate_normality_representation_contract_rejects_rank_mismatch() -> None:
    representation = SimpleNamespace(space="feature")
    probe_output = _representation_output(tensor=torch.ones(4))

    with pytest.raises(ValueError, match="requires representation tensor rank"):
        validate_normality_representation_contract(_FeatureOfflineNormality(), representation, probe_output)


def test_validate_normality_representation_contract_rejects_grad_carrying_offline_output() -> None:
    representation = SimpleNamespace(space="feature")
    tensor = torch.ones(4, 2, 2, requires_grad=True)
    probe_output = _representation_output(tensor=tensor, requires_grad=True)

    with pytest.raises(ValueError, match="detached representations for offline fit mode"):
        validate_normality_representation_contract(_FeatureOfflineNormality(), representation, probe_output)
