"""Base helpers for normality models."""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from typing import Any

import torch

from adrf.core.interfaces import NormalityModel
from adrf.representation.contracts import RepresentationBatch, RepresentationOutput


class BaseNormalityModel(NormalityModel, ABC):
    """Common tensor extraction helpers for normality models."""

    fit_mode = "offline"
    accepted_spaces: frozenset[str] = frozenset()
    accepted_tensor_ranks: frozenset[int] = frozenset()
    requires_detached_representation = True

    def validate_representation_output(self, representation: RepresentationOutput) -> None:
        """Validate a single typed representation against this normality model."""

        if not isinstance(representation, RepresentationOutput):
            raise TypeError(
                f"{type(self).__name__} expects RepresentationOutput inputs, got {type(representation).__name__}."
            )
        representation.validate()
        self._validate_representation_metadata(
            space=representation.space,
            tensor=representation.tensor,
            requires_grad=representation.requires_grad,
        )

    def validate_representation_batch(self, representations: RepresentationBatch) -> None:
        """Validate a typed representation batch against this normality model."""

        if not isinstance(representations, RepresentationBatch):
            raise TypeError(
                f"{type(self).__name__} expects RepresentationBatch inputs, got {type(representations).__name__}."
            )
        representations.validate()
        for representation in representations.unbind():
            self.validate_representation_output(representation)

    def require_representation_tensor(self, representation: RepresentationOutput | Mapping[str, Any]) -> torch.Tensor:
        """Return the validated typed tensor or fall back to the legacy mapping contract."""

        if isinstance(representation, RepresentationOutput):
            self.validate_representation_output(representation)
            return representation.tensor
        if isinstance(representation, Mapping):
            normalized = self._normalize_legacy_representation(representation)
            self._validate_representation_metadata(
                space=str(normalized["space"]),
                tensor=normalized["tensor"],
                requires_grad=bool(normalized["requires_grad"]),
            )
            return normalized["tensor"]
        raise TypeError(
            f"{type(self).__name__} expects RepresentationOutput or mapping inputs, got {type(representation).__name__}."
        )

    @staticmethod
    def serialize_representation(representation: RepresentationOutput | Mapping[str, Any]) -> dict[str, object]:
        """Serialize typed outputs via the contract and preserve legacy mapping payloads."""

        if isinstance(representation, RepresentationOutput):
            return representation.to_artifact_dict()
        if isinstance(representation, Mapping):
            return BaseNormalityModel._normalize_legacy_representation(representation)
        raise TypeError(
            f"Normality models expect RepresentationOutput or mapping payloads, got {type(representation).__name__}."
        )

    def _validate_representation_metadata(
        self,
        *,
        space: str,
        tensor: torch.Tensor,
        requires_grad: bool,
    ) -> None:
        if self.accepted_spaces and space not in self.accepted_spaces:
            accepted = ", ".join(f"`{candidate}`" for candidate in sorted(self.accepted_spaces))
            raise ValueError(f"{type(self).__name__} requires representation space {accepted}; got `{space}`.")
        if self.accepted_tensor_ranks and tensor.ndim not in self.accepted_tensor_ranks:
            accepted = ", ".join(str(rank) for rank in sorted(self.accepted_tensor_ranks))
            raise ValueError(
                f"{type(self).__name__} requires representation tensor rank in {{{accepted}}}; got {tensor.ndim}."
            )
        if self.requires_detached_representation and requires_grad:
            raise ValueError(f"{type(self).__name__} requires detached representations for offline normality.")

    @staticmethod
    def _normalize_legacy_representation(representation: Mapping[str, Any]) -> dict[str, object]:
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
        spatial_shape = tuple(raw_spatial_shape) if raw_spatial_shape is not None else (tuple(tensor.shape[-2:]) if tensor.ndim == 3 else None)

        raw_feature_dim = representation.get("feature_dim")
        feature_dim = int(raw_feature_dim) if raw_feature_dim is not None else int(tensor.shape[0])

        raw_space = representation.get("space", representation.get("space_type", representation.get("representation_space")))
        space = str(raw_space) if raw_space is not None else ""

        return {
            "tensor": tensor,
            "space": space,
            "spatial_shape": spatial_shape,
            "feature_dim": feature_dim,
            "sample_id": representation.get("sample_id"),
            "requires_grad": requires_grad,
            "device": str(representation.get("device")) if representation.get("device") is not None else str(tensor.device),
            "dtype": str(representation.get("dtype")) if representation.get("dtype") is not None else str(tensor.dtype),
            "provenance": provenance_value,
        }
