from __future__ import annotations

import torch

from adrf.normality.representation import (
    NormalizedLegacyRepresentation,
    normalize_normality_representation_input,
    serialize_normality_representation,
)
from adrf.representation.contracts import RepresentationProvenance


def test_normalize_normality_representation_input_returns_typed_output_unchanged() -> None:
    provenance = RepresentationProvenance(
        representation_name="pixel",
        backbone_name="toy",
        weights_source=None,
        feature_layer=None,
        pooling=None,
        trainable=False,
        frozen_submodules=(),
        input_image_size=(16, 16),
        input_normalize=False,
        normalize_mean=None,
        normalize_std=None,
        code_version="tests",
        config_fingerprint="normality-representation",
    )
    from adrf.representation.contracts import RepresentationOutput

    output = RepresentationOutput(
        tensor=torch.ones(3, 16, 16),
        space="pixel",
        spatial_shape=(16, 16),
        feature_dim=3,
        sample_id="sample-001",
        requires_grad=False,
        device="cpu",
        dtype="torch.float32",
        provenance=provenance,
    )

    assert normalize_normality_representation_input(output) is output
    assert serialize_normality_representation(output) == output.to_artifact_dict()


def test_normalize_normality_representation_input_explicitly_adapts_legacy_mapping() -> None:
    legacy = {
        "representation": torch.ones(4, 2, 2),
        "space_type": "feature",
        "spatial_shape": (2, 2),
        "feature_dim": 4,
    }

    normalized = normalize_normality_representation_input(legacy)

    assert normalized == NormalizedLegacyRepresentation(
        tensor=legacy["representation"],
        space="feature",
        spatial_shape=(2, 2),
        feature_dim=4,
        sample_id=None,
        requires_grad=False,
        device="cpu",
        dtype="torch.float32",
        provenance=None,
    )
    assert serialize_normality_representation(legacy) == {
        "tensor": legacy["representation"],
        "space": "feature",
        "spatial_shape": (2, 2),
        "feature_dim": 4,
        "sample_id": None,
        "requires_grad": False,
        "device": "cpu",
        "dtype": "torch.float32",
        "provenance": None,
    }
