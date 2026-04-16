"""Tests for internal diffusion conditioning helpers."""

from __future__ import annotations

import torch

from adrf.core.sample import Sample
from adrf.normality.diffusion_conditioning import (
    DiffusionConditioning,
    build_class_conditioning,
    build_reference_conditioning,
    combine_conditioning,
    resolve_optional_class_ids,
)


def test_build_reference_conditioning_wraps_reference_tensor_and_metadata() -> None:
    conditioning = build_reference_conditioning(torch.ones(3, 4, 4), metadata={"kind": "reference"})

    assert isinstance(conditioning, DiffusionConditioning)
    assert conditioning.reference.shape == (3, 4, 4)
    assert conditioning.metadata == {"kind": "reference"}


def test_build_class_conditioning_wraps_class_ids() -> None:
    conditioning = build_class_conditioning(torch.tensor([1, 2], dtype=torch.long), metadata={"kind": "class"})

    assert isinstance(conditioning, DiffusionConditioning)
    assert torch.equal(conditioning.class_ids, torch.tensor([1, 2], dtype=torch.long))
    assert conditioning.metadata == {"kind": "class"}


def test_combine_conditioning_merges_reference_class_and_metadata() -> None:
    reference = build_reference_conditioning(torch.ones(3, 4, 4), metadata={"reference": True})
    classes = build_class_conditioning(torch.tensor([2], dtype=torch.long), metadata={"class": True})

    merged = combine_conditioning(reference, classes)

    assert merged.reference.shape == (3, 4, 4)
    assert torch.equal(merged.class_ids, torch.tensor([2], dtype=torch.long))
    assert merged.metadata == {"reference": True, "class": True}


def test_resolve_optional_class_ids_returns_none_without_class_configuration() -> None:
    samples = [Sample(image=torch.zeros(3, 4, 4), sample_id="sample-001", category="bottle")]

    resolved = resolve_optional_class_ids(
        samples,
        num_classes=None,
        class_to_index={},
        fit=True,
        backend="legacy",
        model_name="DiffusionBasicNormality",
    )

    assert resolved is None


def test_resolve_optional_class_ids_updates_mapping_for_legacy_fit() -> None:
    samples = [
        Sample(image=torch.zeros(3, 4, 4), sample_id="sample-001", category="bottle"),
        Sample(image=torch.zeros(3, 4, 4), sample_id="sample-002", category="capsule"),
    ]
    class_to_index: dict[str, int] = {}

    resolved = resolve_optional_class_ids(
        samples,
        num_classes=2,
        class_to_index=class_to_index,
        fit=True,
        backend="legacy",
        model_name="DiffusionBasicNormality",
    )

    assert torch.equal(resolved, torch.tensor([0, 1], dtype=torch.long))
    assert class_to_index == {"bottle": 0, "capsule": 1}


def test_resolve_optional_class_ids_rejects_unsupported_backend_explicitly() -> None:
    samples = [Sample(image=torch.zeros(3, 4, 4), sample_id="sample-001", category="bottle")]

    try:
        resolve_optional_class_ids(
            samples,
            num_classes=2,
            class_to_index={},
            fit=True,
            backend="diffusers",
            model_name="DiffusionBasicNormality",
        )
    except NotImplementedError as error:
        assert "Class-conditioned diffusers backend is not implemented yet." in str(error)
    else:
        raise AssertionError("resolve_optional_class_ids should reject unsupported class-conditioned backends.")
