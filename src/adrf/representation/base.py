"""Base helpers for representation models."""

from __future__ import annotations

from abc import ABC

import torch

from adrf.core.interfaces import Representation
from adrf.core.sample import Sample


class BaseRepresentation(Representation, ABC):
    """Small base class for representations operating on image tensors."""

    @staticmethod
    def require_image_tensor(sample: Sample) -> torch.Tensor:
        """Return the sample image tensor or raise when the sample is not transformed."""

        image = sample.image
        if not isinstance(image, torch.Tensor):
            raise TypeError("Representation models expect sample.image to be a torch.Tensor.")
        if image.ndim != 3:
            raise TypeError("Representation models expect sample.image with shape [C, H, W].")
        return image

