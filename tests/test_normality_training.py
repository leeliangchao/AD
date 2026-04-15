from __future__ import annotations

import pytest
from torch import nn

from adrf.normality.training import (
    JointNormalityTrainingAdapter,
    OfflineNormalityTrainingAdapter,
    resolve_normality_training_adapter,
)


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


class _IncompleteJointNormality:
    fit_mode = "joint"


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


def test_resolve_normality_training_adapter_rejects_incomplete_joint_contract() -> None:
    with pytest.raises(RuntimeError, match="must implement joint training"):
        resolve_normality_training_adapter(_IncompleteJointNormality())
