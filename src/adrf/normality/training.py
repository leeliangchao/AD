from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from adrf.core.interfaces import NormalityModel


class BaseNormalityTrainingAdapter(ABC):
    mode: str

    def __init__(self, normality: object) -> None:
        self.normality = normality


class OfflineNormalityTrainingAdapter(BaseNormalityTrainingAdapter):
    mode = "offline"

    def fit(self, representations, samples=None) -> None:
        self.normality.fit(representations, samples)  # type: ignore[attr-defined]


class JointNormalityTrainingAdapter(BaseNormalityTrainingAdapter):
    mode = "joint"

    def configure_joint_training(self, representation_model: Any) -> None:
        self.normality.configure_joint_training(representation_model)  # type: ignore[attr-defined]

    def fit_batch(self, representations, samples) -> dict[str, float]:
        return self.normality.fit_batch(representations, samples)  # type: ignore[attr-defined]


def resolve_normality_training_adapter(normality: object) -> BaseNormalityTrainingAdapter:
    fit_mode = str(getattr(normality, "fit_mode", "offline"))
    if fit_mode == "offline":
        return OfflineNormalityTrainingAdapter(normality)
    if fit_mode == "joint":
        _validate_joint_training_contract(normality)
        return JointNormalityTrainingAdapter(normality)
    raise ValueError(f"Unsupported normality fit mode: {fit_mode}")


def _validate_joint_training_contract(normality: object) -> None:
    configure_joint = getattr(normality, "configure_joint_training", None)
    fit_batch = getattr(normality, "fit_batch", None)
    if not callable(configure_joint) or getattr(type(normality), "configure_joint_training", None) is getattr(
        NormalityModel,
        "configure_joint_training",
        None,
    ):
        raise RuntimeError(f"{type(normality).__name__} must implement joint training configure_joint_training().")
    if not callable(fit_batch) or getattr(type(normality), "fit_batch", None) is getattr(
        NormalityModel,
        "fit_batch",
        None,
    ):
        raise RuntimeError(f"{type(normality).__name__} must implement joint training fit_batch().")
