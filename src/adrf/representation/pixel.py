"""Pixel-space representation model."""

from __future__ import annotations

import torch

from adrf.representation.base import BaseRepresentation
from adrf.representation.contracts import RepresentationProvenance


class PixelRepresentation(BaseRepresentation):
    """Expose the transformed image tensor directly as the pixel representation."""

    space = "pixel"
    trainable = False

    def __init__(self, input_image_size: tuple[int, int] = (256, 256), input_normalize: bool = False) -> None:
        super().__init__(input_image_size=input_image_size, input_normalize=input_normalize)

    def _encode_tensor_batch(self, batch: torch.Tensor) -> torch.Tensor:
        return batch

    def describe(self) -> RepresentationProvenance:
        return RepresentationProvenance(
            representation_name="pixel",
            backbone_name=None,
            weights_source=None,
            feature_layer=None,
            pooling=None,
            trainable=False,
            frozen_submodules=(),
            input_image_size=self.input_image_size,
            input_normalize=self.input_normalize,
            normalize_mean=None,
            normalize_std=None,
            code_version="working-tree",
            config_fingerprint="pixel",
        )
