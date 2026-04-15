"""Base helpers for normality models."""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from typing import Any

import torch

from adrf.normality.representation import (
    NormalizedLegacyRepresentation,
    normalize_normality_representation_input,
    serialize_normality_representation,
)
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
        """Return the validated typed tensor or one explicit legacy-normalized tensor."""

        normalized = normalize_normality_representation_input(representation)
        if isinstance(normalized, RepresentationOutput):
            self.validate_representation_output(normalized)
            return normalized.tensor
        self._validate_legacy_representation_output(normalized)
        return normalized.tensor

    @staticmethod
    def serialize_representation(representation: RepresentationOutput | Mapping[str, Any]) -> dict[str, object]:
        """Serialize typed outputs via the contract and preserve legacy payload compatibility explicitly."""

        return serialize_normality_representation(representation)

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

    def _validate_legacy_representation_output(self, representation: NormalizedLegacyRepresentation) -> None:
        self._validate_representation_metadata(
            space=representation.space,
            tensor=representation.tensor,
            requires_grad=representation.requires_grad,
        )
