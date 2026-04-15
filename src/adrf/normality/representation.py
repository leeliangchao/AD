from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from adrf.representation.contracts import RepresentationOutput


@dataclass(frozen=True, slots=True)
class NormalizedLegacyRepresentation:
    tensor: torch.Tensor
    space: str
    spatial_shape: tuple[int, int] | None
    feature_dim: int
    sample_id: str | None
    requires_grad: bool
    device: str
    dtype: str
    provenance: dict[str, Any] | None

    def to_artifact_dict(self) -> dict[str, object]:
        return {
            "tensor": self.tensor,
            "space": self.space,
            "spatial_shape": self.spatial_shape,
            "feature_dim": self.feature_dim,
            "sample_id": self.sample_id,
            "requires_grad": self.requires_grad,
            "device": self.device,
            "dtype": self.dtype,
            "provenance": self.provenance,
        }


def normalize_normality_representation_input(
    representation: RepresentationOutput | Mapping[str, Any],
) -> RepresentationOutput | NormalizedLegacyRepresentation:
    if isinstance(representation, RepresentationOutput):
        representation.validate()
        return representation
    if isinstance(representation, Mapping):
        return _normalize_legacy_representation(representation)
    raise TypeError(
        "Normality models expect RepresentationOutput or mapping inputs, "
        f"got {type(representation).__name__}."
    )


def serialize_normality_representation(
    representation: RepresentationOutput | Mapping[str, Any],
) -> dict[str, object]:
    normalized = normalize_normality_representation_input(representation)
    if isinstance(normalized, RepresentationOutput):
        return normalized.to_artifact_dict()
    return normalized.to_artifact_dict()


def rehome_normality_representation(
    representation: RepresentationOutput | Mapping[str, Any],
    device: torch.device,
) -> RepresentationOutput | NormalizedLegacyRepresentation:
    normalized = normalize_normality_representation_input(representation)
    moved_tensor = normalized.tensor.to(device)
    if isinstance(normalized, RepresentationOutput):
        return RepresentationOutput(
            tensor=moved_tensor,
            space=normalized.space,
            spatial_shape=normalized.spatial_shape,
            feature_dim=normalized.feature_dim,
            sample_id=normalized.sample_id,
            requires_grad=bool(moved_tensor.requires_grad),
            device=str(moved_tensor.device),
            dtype=str(moved_tensor.dtype),
            provenance=normalized.provenance,
        )
    return NormalizedLegacyRepresentation(
        tensor=moved_tensor,
        space=normalized.space,
        spatial_shape=normalized.spatial_shape,
        feature_dim=normalized.feature_dim,
        sample_id=normalized.sample_id,
        requires_grad=bool(moved_tensor.requires_grad),
        device=str(moved_tensor.device),
        dtype=str(moved_tensor.dtype),
        provenance=normalized.provenance,
    )


def _normalize_legacy_representation(representation: Mapping[str, Any]) -> NormalizedLegacyRepresentation:
    tensor = representation.get("tensor", representation.get("representation"))
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Normality models expect representation['representation'] to be a torch.Tensor.")

    raw_requires_grad = representation.get("requires_grad")
    requires_grad = tensor.requires_grad if raw_requires_grad is None else bool(raw_requires_grad) or tensor.requires_grad

    provenance = representation.get("provenance")
    if hasattr(provenance, "to_dict"):
        provenance_value = provenance.to_dict()
    elif isinstance(provenance, Mapping):
        provenance_value = dict(provenance)
    else:
        provenance_value = None

    raw_spatial_shape = representation.get("spatial_shape")
    spatial_shape = (
        tuple(raw_spatial_shape)
        if raw_spatial_shape is not None
        else (tuple(tensor.shape[-2:]) if tensor.ndim == 3 else None)
    )

    raw_feature_dim = representation.get("feature_dim")
    feature_dim = int(raw_feature_dim) if raw_feature_dim is not None else int(tensor.shape[0])

    raw_space = representation.get("space", representation.get("space_type", representation.get("representation_space")))
    space = str(raw_space) if raw_space is not None else ""

    return NormalizedLegacyRepresentation(
        tensor=tensor,
        space=space,
        spatial_shape=spatial_shape,
        feature_dim=feature_dim,
        sample_id=representation.get("sample_id"),
        requires_grad=requires_grad,
        device=str(representation.get("device")) if representation.get("device") is not None else str(tensor.device),
        dtype=str(representation.get("dtype")) if representation.get("dtype") is not None else str(tensor.dtype),
        provenance=provenance_value,
    )
