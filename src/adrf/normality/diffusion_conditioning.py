"""Internal helpers for diffusion conditioning payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class DiffusionConditioning:
    """Canonical internal representation for optional diffusion conditioning inputs."""

    reference: torch.Tensor | None = None
    class_ids: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def build_reference_conditioning(
    reference: torch.Tensor,
    *,
    metadata: dict[str, Any] | None = None,
) -> DiffusionConditioning:
    return DiffusionConditioning(reference=reference, metadata=dict(metadata or {}))


def build_class_conditioning(
    class_ids: torch.Tensor,
    *,
    metadata: dict[str, Any] | None = None,
) -> DiffusionConditioning:
    return DiffusionConditioning(class_ids=class_ids, metadata=dict(metadata or {}))


def combine_conditioning(*conditionings: DiffusionConditioning) -> DiffusionConditioning:
    reference = None
    class_ids = None
    metadata: dict[str, Any] = {}
    for conditioning in conditionings:
        if conditioning.reference is not None:
            reference = conditioning.reference
        if conditioning.class_ids is not None:
            class_ids = conditioning.class_ids
        metadata.update(conditioning.metadata)
    return DiffusionConditioning(reference=reference, class_ids=class_ids, metadata=metadata)
