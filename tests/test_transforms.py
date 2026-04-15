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


def test_sample_transform_resizes_and_tensorizes_reference() -> None:
    """The transform should convert reference inputs into aligned tensors."""

    transform = SampleTransform(image_size=(16, 12), normalize=False)
    sample = Sample(
        image=Image.new("RGB", (8, 10), color=(255, 0, 0)),
        reference=Image.new("RGB", (8, 10), color=(0, 255, 0)),
    )

    transformed = transform(sample)

    assert isinstance(transformed.reference, torch.Tensor)
    assert transformed.reference.shape == (3, 16, 12)


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


def test_sample_transform_normalizes_reference_tensor_values() -> None:
    """Reference tensors should follow the same normalization policy as images."""

    transform = SampleTransform(
        image_size=(4, 4),
        normalize=True,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )
    sample = Sample(
        image=Image.new("RGB", (4, 4), color=(255, 255, 255)),
        reference=Image.new("RGB", (4, 4), color=(255, 255, 255)),
    )

    transformed = transform(sample)

    assert torch.isclose(transformed.reference.min(), torch.tensor(1.0))
    assert torch.isclose(transformed.reference.max(), torch.tensor(1.0))


def test_sample_transform_treats_pil_and_uint8_tensor_images_equally() -> None:
    """Equivalent PIL and uint8 tensor images should produce the same float tensor."""

    transform = SampleTransform(image_size=(2, 2), normalize=False)
    rgb = (255, 128, 0)
    pil_sample = Sample(image=Image.new("RGB", (2, 2), color=rgb))
    tensor_sample = Sample(image=torch.tensor(rgb, dtype=torch.uint8).view(3, 1, 1).expand(3, 2, 2))

    pil_transformed = transform(pil_sample)
    tensor_transformed = transform(tensor_sample)

    assert torch.allclose(pil_transformed.image, tensor_transformed.image)
