"""Tests for pixel and feature representations."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from adrf.core.sample import Sample
from adrf.representation.base import BaseRepresentation
from adrf.representation.contracts import RepresentationBatch, RepresentationProvenance
from adrf.representation.feature import FeatureRepresentation
from adrf.representation.pixel import PixelRepresentation


class _BuggyRepresentation(BaseRepresentation):
    space = "pixel"

    def __init__(self, batch_size: int) -> None:
        super().__init__(input_image_size=(4, 4), input_normalize=False)
        self._batch_size = batch_size

    def encode_batch(self, samples: Sequence[Sample]) -> RepresentationBatch:
        del samples
        tensor = torch.zeros((self._batch_size, 3, 4, 4))
        return RepresentationBatch(
            tensor=tensor,
            space=self.space,
            spatial_shape=(4, 4),
            feature_dim=3,
            batch_size=self._batch_size,
            sample_ids=tuple(f"buggy-{index}" for index in range(self._batch_size)),
            requires_grad=False,
            device=str(tensor.device),
            dtype=str(tensor.dtype),
            provenance=self.describe(),
        )

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
            code_version="tests",
            config_fingerprint=f"buggy:{self._batch_size}",
        )


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


def test_feature_representation_legacy_constructor_kwargs_map_to_new_contract() -> None:
    model = FeatureRepresentation(pretrained=False, freeze=True, input_image_size=(64, 64), input_normalize=False)

    assert model.weights is None
    assert model.trainable is False
    assert all(parameter.requires_grad is False for parameter in model.backbone.parameters())


def test_representation_legacy_callable_returns_mapping_payload() -> None:
    pixel_payload = PixelRepresentation(input_image_size=(16, 20), input_normalize=False)(
        Sample(image=torch.rand(3, 16, 20), sample_id="pixel-legacy")
    )
    feature_payload = FeatureRepresentation(pretrained=False, freeze=True, input_image_size=(64, 64), input_normalize=False)(
        Sample(image=torch.rand(3, 64, 64), sample_id="feature-legacy")
    )

    assert pixel_payload["space_type"] == "pixel"
    assert pixel_payload["representation"].shape == (3, 16, 20)
    assert pixel_payload["spatial_shape"] == (16, 20)
    assert feature_payload["space_type"] == "feature"
    assert feature_payload["representation"].ndim == 3
    assert feature_payload["feature_dim"] == feature_payload["representation"].shape[0]
    assert feature_payload["spatial_shape"] == tuple(feature_payload["representation"].shape[-2:])


def test_feature_representation_rejects_unsupported_weights() -> None:
    with pytest.raises(ValueError, match="unsupported weights"):
        FeatureRepresentation(weights="custom-backbone", input_image_size=(64, 64), input_normalize=False)


@pytest.mark.parametrize("batch_size", [0, 2])
def test_base_representation_encode_sample_rejects_invalid_output_cardinality(batch_size: int) -> None:
    representation = _BuggyRepresentation(batch_size=batch_size)

    with pytest.raises(
        ValueError,
        match=rf"_BuggyRepresentation.encode_sample expected exactly one output for one input sample, got batch_size={batch_size}\.",
    ):
        representation.encode_sample(Sample(image=torch.rand(3, 4, 4), sample_id="buggy"))
