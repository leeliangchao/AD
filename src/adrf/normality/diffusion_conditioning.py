"""Internal helpers for diffusion conditioning payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from typing import Sequence

import torch

from adrf.core.sample import Sample


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


def resolve_class_ids_from_samples(
    samples: list[Sample],
    *,
    num_classes: int,
    class_to_index: dict[str, int],
    fit: bool,
) -> torch.Tensor:
    """Resolve class ids from metadata or category strings, updating the mapping during fit."""

    indices: list[int] = []
    for sample in samples:
        metadata_class_id = sample.metadata.get("class_id")
        if isinstance(metadata_class_id, int):
            class_id = metadata_class_id
        elif sample.category is not None:
            if sample.category not in class_to_index:
                if not fit:
                    raise ValueError(f"Unknown category `{sample.category}` for class-conditioned diffusion inference.")
                if len(class_to_index) >= num_classes:
                    raise ValueError("Observed more categories than configured num_classes.")
                class_to_index[sample.category] = len(class_to_index)
            class_id = class_to_index[sample.category]
        else:
            raise ValueError("Class-conditioned diffusion requires `sample.metadata['class_id']` or `sample.category`.")
        if not 0 <= class_id < num_classes:
            raise ValueError(f"class_id={class_id} is outside configured range [0, {num_classes}).")
        indices.append(class_id)
    return torch.tensor(indices, dtype=torch.long)


def resolve_optional_class_ids(
    samples: list[Sample] | None,
    *,
    num_classes: int | None,
    class_to_index: dict[str, int],
    fit: bool,
    backend: str,
    supported_backends: Sequence[str] = ("legacy",),
    model_name: str,
) -> torch.Tensor | None:
    """Resolve class ids when class conditioning is enabled and supported."""

    if num_classes is None:
        return None
    if samples is None:
        raise ValueError(f"{model_name} class conditioning requires samples.")
    normalized_supported_backends = {value.strip().lower() for value in supported_backends}
    if backend.strip().lower() not in normalized_supported_backends:
        raise NotImplementedError("Class-conditioned diffusers backend is not implemented yet.")
    return resolve_class_ids_from_samples(
        samples,
        num_classes=num_classes,
        class_to_index=class_to_index,
        fit=fit,
    )
