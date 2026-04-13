"""Tests for pixel and feature representations."""

import torch

from adrf.core.sample import Sample
from adrf.representation.feature import FeatureRepresentation
from adrf.representation.pixel import PixelRepresentation


def test_pixel_representation_returns_pixel_space_metadata() -> None:
    """Pixel representation should preserve the input tensor and report shape."""

    sample = Sample(image=torch.rand(3, 16, 20))

    representation = PixelRepresentation()(sample)

    assert representation["space_type"] == "pixel"
    assert representation["representation"].shape == (3, 16, 20)
    assert representation["spatial_shape"] == (16, 20)


def test_feature_representation_returns_feature_map_and_freezes_backbone() -> None:
    """Feature representation should emit a feature tensor with stable metadata."""

    model = FeatureRepresentation(pretrained=False, freeze=True)
    sample = Sample(image=torch.rand(3, 64, 64))

    representation = model(sample)

    assert representation["space_type"] == "feature"
    assert representation["representation"].ndim == 3
    assert representation["feature_dim"] == representation["representation"].shape[0]
    assert representation["spatial_shape"] == tuple(representation["representation"].shape[-2:])
    assert all(parameter.requires_grad is False for parameter in model.backbone.parameters())

