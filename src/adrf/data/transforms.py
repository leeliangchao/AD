"""Minimal transforms for MVTec images and masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tv_functional

from adrf.core.sample import Sample


@dataclass(slots=True)
class SampleTransform:
    """Resize, tensorize, and optionally normalize a sample image and mask."""

    image_size: tuple[int, int] = (256, 256)
    normalize: bool = True
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __call__(self, sample: Sample) -> Sample:
        """Return a transformed copy of the provided sample."""

        image = self._transform_image(sample.image)
        mask = self._transform_mask(sample.mask)
        reference = self._transform_reference(sample.reference)
        return Sample(
            image=image,
            label=sample.label,
            mask=mask,
            category=sample.category,
            sample_id=sample.sample_id,
            reference=reference,
            views=dict(sample.views) if sample.views is not None else None,
            metadata=dict(sample.metadata),
        )

    def _transform_image(self, image: Image.Image | torch.Tensor) -> torch.Tensor:
        """Resize and convert an image into a normalized float tensor."""

        if isinstance(image, Image.Image):
            resized = tv_functional.resize(
                image,
                self.image_size,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            tensor = tv_functional.to_tensor(resized)
        elif isinstance(image, torch.Tensor):
            tensor = image.float()
            if tensor.ndim == 3:
                tensor = tv_functional.resize(
                    tensor,
                    self.image_size,
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )
            else:
                raise TypeError("Sample image tensor must have shape [C, H, W].")
        else:
            raise TypeError("Sample image must be a PIL image or torch tensor.")

        if self.normalize:
            tensor = tv_functional.normalize(tensor, self.mean, self.std)
        return tensor

    def _transform_reference(self, reference: Image.Image | torch.Tensor | None) -> torch.Tensor | None:
        """Resize and convert a reference image using the same policy as the sample image."""

        if reference is None:
            return None
        return self._transform_image(reference)

    def _transform_mask(self, mask: Image.Image | torch.Tensor | None) -> torch.Tensor | None:
        """Resize and convert a mask into a float tensor when present."""

        if mask is None:
            return None

        if isinstance(mask, Image.Image):
            resized = tv_functional.resize(
                mask,
                self.image_size,
                interpolation=InterpolationMode.NEAREST,
            )
            tensor = tv_functional.pil_to_tensor(resized).float() / 255.0
        elif isinstance(mask, torch.Tensor):
            tensor = mask.float()
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 3:
                raise TypeError("Sample mask tensor must have shape [H, W] or [1, H, W].")
            tensor = tv_functional.resize(
                tensor,
                self.image_size,
                interpolation=InterpolationMode.NEAREST,
                antialias=False,
            )
            if tensor.max() > 1:
                tensor = tensor / 255.0
        else:
            raise TypeError("Sample mask must be a PIL image, torch tensor, or None.")

        return cast(torch.Tensor, tensor.clamp(0.0, 1.0))


def build_sample_transform(
    image_size: tuple[int, int] = (256, 256),
    normalize: bool = True,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> SampleTransform:
    """Construct the default sample transform for MVTec inputs."""

    return SampleTransform(image_size=image_size, normalize=normalize, mean=mean, std=std)
