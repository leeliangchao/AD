"""Tests for internal diffusion conditioning helpers."""

from __future__ import annotations

import torch

from adrf.normality.diffusion_conditioning import (
    DiffusionConditioning,
    build_class_conditioning,
    build_reference_conditioning,
    combine_conditioning,
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
