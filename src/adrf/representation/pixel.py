"""Pixel-space representation model."""

from __future__ import annotations

from typing import Any

from adrf.core.sample import Sample
from adrf.representation.base import BaseRepresentation


class PixelRepresentation(BaseRepresentation):
    """Expose the transformed image tensor directly as the pixel representation."""

    def __call__(self, sample: Sample) -> dict[str, Any]:
        """Return the sample image in pixel space."""

        image = self.require_image_tensor(sample)
        return {
            "representation": image,
            "space_type": "pixel",
            "spatial_shape": tuple(image.shape[-2:]),
        }

