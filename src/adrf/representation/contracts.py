from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class RepresentationProvenance:
    representation_name: str
    backbone_name: str | None
    weights_source: str | None
    feature_layer: str | None
    pooling: str | None
    trainable: bool
    frozen_submodules: tuple[str, ...]
    input_image_size: tuple[int, int] | None
    input_normalize: bool
    normalize_mean: tuple[float, ...] | None
    normalize_std: tuple[float, ...] | None
    code_version: str
    config_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation_name": self.representation_name,
            "backbone_name": self.backbone_name,
            "weights_source": self.weights_source,
            "feature_layer": self.feature_layer,
            "pooling": self.pooling,
            "trainable": self.trainable,
            "frozen_submodules": list(self.frozen_submodules),
            "input_image_size": list(self.input_image_size) if self.input_image_size is not None else None,
            "input_normalize": self.input_normalize,
            "normalize_mean": list(self.normalize_mean) if self.normalize_mean is not None else None,
            "normalize_std": list(self.normalize_std) if self.normalize_std is not None else None,
            "code_version": self.code_version,
            "config_fingerprint": self.config_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class RepresentationOutput:
    tensor: torch.Tensor
    space: str
    spatial_shape: tuple[int, int] | None
    feature_dim: int
    sample_id: str | None
    requires_grad: bool
    device: str
    dtype: str
    provenance: RepresentationProvenance

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "tensor": self.tensor,
            "space": self.space,
            "spatial_shape": self.spatial_shape,
            "feature_dim": self.feature_dim,
            "sample_id": self.sample_id,
            "requires_grad": self.requires_grad,
            "device": self.device,
            "dtype": self.dtype,
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RepresentationBatch:
    tensor: torch.Tensor
    space: str
    spatial_shape: tuple[int, int] | None
    feature_dim: int
    batch_size: int
    sample_ids: tuple[str | None, ...]
    requires_grad: bool
    device: str
    dtype: str
    provenance: RepresentationProvenance

    def unbind(self) -> list[RepresentationOutput]:
        if self.tensor.shape[0] != self.batch_size:
            raise ValueError(
                f"RepresentationBatch batch_size={self.batch_size} does not match tensor batch dimension "
                f"{self.tensor.shape[0]}."
            )
        if len(self.sample_ids) != self.batch_size:
            raise ValueError("RepresentationBatch sample_ids length must match batch_size.")
        return [
            RepresentationOutput(
                tensor=self.tensor[index],
                space=self.space,
                spatial_shape=self.spatial_shape,
                feature_dim=self.feature_dim,
                sample_id=self.sample_ids[index],
                requires_grad=self.requires_grad,
                device=self.device,
                dtype=self.dtype,
                provenance=self.provenance,
            )
            for index in range(self.batch_size)
        ]
