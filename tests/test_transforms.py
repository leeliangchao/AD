"""Tests for minimal image and mask transforms."""

import torch
from PIL import Image

from adrf.core.sample import Sample
from adrf.data.transforms import SampleTransform


def test_sample_transform_resizes_and_tensorizes_image_and_mask() -> None:
    """The transform should convert PIL inputs into tensors with target size."""

    transform = SampleTransform(image_size=(16, 12), normalize=False)
    sample = Sample(
        image=Image.new("RGB", (8, 10), color=(255, 0, 0)),
        mask=Image.new("L", (8, 10), color=255),
    )

    transformed = transform(sample)

    assert isinstance(transformed.image, torch.Tensor)
    assert isinstance(transformed.mask, torch.Tensor)
    assert transformed.image.shape == (3, 16, 12)
    assert transformed.mask.shape == (1, 16, 12)
    assert torch.isclose(transformed.mask.max(), torch.tensor(1.0))


def test_sample_transform_normalizes_image_tensor_values() -> None:
    """Normalization should be applied after tensor conversion."""

    transform = SampleTransform(
        image_size=(4, 4),
        normalize=True,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )
    sample = Sample(image=Image.new("RGB", (4, 4), color=(255, 255, 255)))

    transformed = transform(sample)

    assert torch.isclose(transformed.image.min(), torch.tensor(1.0))
    assert torch.isclose(transformed.image.max(), torch.tensor(1.0))

