from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from adrf.core.interfaces import NormalityModel
from adrf.representation.contracts import RepresentationOutput


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


def validate_normality_representation_contract(
    normality: object,
    representation: object,
    probe_output: RepresentationOutput,
) -> None:
    representation_space = getattr(representation, "space", None)
    accepted_spaces = getattr(normality, "accepted_spaces", frozenset())
    if accepted_spaces and representation_space not in accepted_spaces:
        accepted = ", ".join(f"`{candidate}`" for candidate in sorted(accepted_spaces))
        raise ValueError(
            f"{type(normality).__name__} requires representation space {accepted}; "
            f"got `{representation_space}` from {type(getattr(representation, 'representation', representation)).__name__}."
        )

    accepted_tensor_ranks = getattr(normality, "accepted_tensor_ranks", frozenset())
    if accepted_tensor_ranks and probe_output.tensor.ndim not in accepted_tensor_ranks:
        accepted = ", ".join(str(rank) for rank in sorted(accepted_tensor_ranks))
        raise ValueError(
            f"{type(normality).__name__} requires representation tensor rank in {{{accepted}}}; "
            f"got {probe_output.tensor.ndim} from {type(getattr(representation, 'representation', representation)).__name__}."
        )

    fit_mode = str(getattr(normality, "fit_mode", "offline"))
    if (
        fit_mode == "offline"
        and bool(getattr(normality, "requires_detached_representation", False))
        and probe_output.requires_grad
    ):
        raise ValueError(
            f"{type(normality).__name__} requires detached representations for offline fit mode, "
            f"but {type(getattr(representation, 'representation', representation)).__name__} emits a trainable representation."
        )

    resolve_normality_training_adapter(normality)
