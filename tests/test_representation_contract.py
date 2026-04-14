from typing import get_args

import torch
import pytest

from adrf.representation import RepresentationSpace
from adrf.representation.contracts import (
    RepresentationBatch,
    RepresentationOutput,
    RepresentationProvenance,
)


def _provenance() -> RepresentationProvenance:
    return RepresentationProvenance(
        representation_name="feature",
        backbone_name="resnet18",
        weights_source="imagenet1k_v1",
        feature_layer="layer4",
        pooling=None,
        trainable=False,
        frozen_submodules=("backbone",),
        input_image_size=(64, 64),
        input_normalize=False,
        normalize_mean=None,
        normalize_std=None,
        code_version="test-sha",
        config_fingerprint="cfg-123",
    )


def test_representation_batch_unbind_preserves_metadata() -> None:
    batch = RepresentationBatch(
        tensor=torch.arange(2 * 4 * 2 * 2, dtype=torch.float32).reshape(2, 4, 2, 2),
        space="feature",
        spatial_shape=(2, 2),
        feature_dim=4,
        batch_size=2,
        sample_ids=("sample-0", "sample-1"),
        requires_grad=False,
        device="cpu",
        dtype="torch.float32",
        provenance=_provenance(),
    )

    outputs = batch.unbind()

    assert [output.sample_id for output in outputs] == ["sample-0", "sample-1"]
    assert outputs[0].space == "feature"
    assert outputs[0].feature_dim == 4
    assert outputs[0].spatial_shape == (2, 2)
    assert outputs[0].provenance.representation_name == "feature"


def test_representation_batch_unbind_rejects_zero_rank_tensor() -> None:
    with pytest.raises(ValueError, match="rank 2"):
        RepresentationBatch(
            tensor=torch.tensor(1.0),
            space="feature",
            spatial_shape=None,
            feature_dim=1,
            batch_size=1,
            sample_ids=("sample-0",),
            requires_grad=False,
            device="cpu",
            dtype="torch.float32",
            provenance=_provenance(),
        )


def test_representation_batch_unbind_rejects_non_spatial_rank_three_tensor() -> None:
    with pytest.raises(ValueError, match="rank 2"):
        RepresentationBatch(
            tensor=torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3),
            space="feature",
            spatial_shape=None,
            feature_dim=4,
            batch_size=2,
            sample_ids=("sample-0", "sample-1"),
            requires_grad=False,
            device="cpu",
            dtype="torch.float32",
            provenance=_provenance(),
        )


def test_representation_batch_unbind_rejects_spatial_rank_five_tensor() -> None:
    with pytest.raises(ValueError, match="rank 4"):
        RepresentationBatch(
            tensor=torch.arange(2 * 4 * 2 * 2 * 2, dtype=torch.float32).reshape(2, 4, 2, 2, 2),
            space="feature",
            spatial_shape=(2, 2),
            feature_dim=4,
            batch_size=2,
            sample_ids=("sample-0", "sample-1"),
            requires_grad=False,
            device="cpu",
            dtype="torch.float32",
            provenance=_provenance(),
        )


def test_representation_batch_unbind_rejects_metadata_mismatch() -> None:
    with pytest.raises(ValueError, match="metadata"):
        RepresentationBatch(
            tensor=torch.arange(2 * 4 * 2 * 2, dtype=torch.float32).reshape(2, 4, 2, 2),
            space="feature",
            spatial_shape=(4, 4),
            feature_dim=8,
            batch_size=2,
            sample_ids=("sample-0", "sample-1"),
            requires_grad=False,
            device="cpu",
            dtype="torch.float32",
            provenance=_provenance(),
        )


def test_representation_space_is_exported_from_representation_package() -> None:
    assert get_args(RepresentationSpace) == ("pixel", "feature")


def test_representation_output_serializes_provenance_for_artifacts() -> None:
    output = RepresentationOutput(
        tensor=torch.ones(4, 2, 2),
        space="feature",
        spatial_shape=(2, 2),
        feature_dim=4,
        sample_id="sample-7",
        requires_grad=False,
        device="cpu",
        dtype="torch.float32",
        provenance=_provenance(),
    )

    payload = output.to_artifact_dict()

    assert payload["space"] == "feature"
    assert payload["sample_id"] == "sample-7"
    assert payload["provenance"]["weights_source"] == "imagenet1k_v1"
