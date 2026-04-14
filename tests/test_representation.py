"""Tests for pixel and feature representations."""

import torch

from adrf.core.sample import Sample
from adrf.representation.feature import FeatureRepresentation
from adrf.representation.pixel import PixelRepresentation


def test_pixel_representation_encode_batch_returns_representation_batch() -> None:
    samples = [
        Sample(image=torch.rand(3, 16, 20), sample_id="pixel-0"),
        Sample(image=torch.rand(3, 16, 20), sample_id="pixel-1"),
    ]

    batch = PixelRepresentation(input_image_size=(16, 20), input_normalize=False).encode_batch(samples)

    assert batch.space == "pixel"
    assert batch.tensor.shape == (2, 3, 16, 20)
    assert batch.batch_size == 2
    assert batch.sample_ids == ("pixel-0", "pixel-1")


def test_feature_representation_encode_sample_reports_feature_metadata() -> None:
    model = FeatureRepresentation(weights=None, trainable=False, input_image_size=(64, 64), input_normalize=False)
    output = model.encode_sample(Sample(image=torch.rand(3, 64, 64), sample_id="feature-0"))

    assert output.space == "feature"
    assert output.tensor.ndim == 3
    assert output.feature_dim == output.tensor.shape[0]
    assert output.provenance.backbone_name == "resnet18"
