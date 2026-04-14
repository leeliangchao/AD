"""Base helpers for representation models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import nn

from adrf.core.interfaces import RepresentationModel
from adrf.core.sample import Sample
from adrf.representation.contracts import RepresentationBatch, RepresentationOutput, RepresentationProvenance


class BaseRepresentation(nn.Module, RepresentationModel, ABC):
    """Base module for batch-aware image representations."""

    space: str
    trainable: bool = False

    def __init__(self, input_image_size: tuple[int, int], input_normalize: bool) -> None:
        super().__init__()
        self.input_image_size = input_image_size
        self.input_normalize = input_normalize

    def encode_sample(self, sample: Sample) -> RepresentationOutput:
        return self.encode_batch([sample]).unbind()[0]

    def encode_batch(self, samples: Sequence[Sample]) -> RepresentationBatch:
        if not samples:
            raise ValueError("encode_batch() requires at least one sample.")

        images = [self.require_image_tensor(sample) for sample in samples]
        batch_tensor = torch.stack(images, dim=0)
        encoded = self._encode_tensor_batch(batch_tensor)
        spatial_shape = tuple(encoded.shape[-2:]) if encoded.ndim == 4 else None
        feature_dim = int(encoded.shape[1] if encoded.ndim >= 2 else encoded.shape[0])
        return RepresentationBatch(
            tensor=encoded,
            space=self.space,
            spatial_shape=spatial_shape,
            feature_dim=feature_dim,
            batch_size=len(samples),
            sample_ids=tuple(sample.sample_id for sample in samples),
            requires_grad=bool(encoded.requires_grad),
            device=str(encoded.device),
            dtype=str(encoded.dtype),
            provenance=self.describe(),
        )

    @staticmethod
    def require_image_tensor(sample: Sample) -> torch.Tensor:
        """Return the sample image tensor or raise when the sample is not transformed."""

        image = sample.image
        if not isinstance(image, torch.Tensor):
            raise TypeError("Representation models expect sample.image to be a torch.Tensor.")
        if image.ndim != 3:
            raise TypeError("Representation models expect sample.image with shape [C, H, W].")
        return image

    @abstractmethod
    def _encode_tensor_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Encode a stacked image batch."""

    @abstractmethod
    def describe(self) -> RepresentationProvenance:
        """Return the representation provenance shared by all outputs."""
