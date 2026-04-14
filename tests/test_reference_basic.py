"""Tests for the minimal reference-conditioned normality model."""

from collections.abc import Iterable
from pathlib import Path
import sys
from typing import get_type_hints

from PIL import Image
import torch

from adrf.core.sample import Sample
from adrf.normality.reference_basic import ReferenceBasicNormality
from adrf.representation.contracts import RepresentationOutput

sys.path.insert(0, str(Path(__file__).parent))

from support.representation_builders import make_pixel_output


def test_reference_basic_public_annotations_expose_representation_output_only() -> None:
    """ReferenceBasicNormality should expose typed RepresentationOutput annotations."""

    fit_hints = get_type_hints(ReferenceBasicNormality.fit)
    infer_hints = get_type_hints(ReferenceBasicNormality.infer)
    helper_hints = get_type_hints(ReferenceBasicNormality._prepare_reference_tensor)

    assert fit_hints["representations"] == Iterable[RepresentationOutput]
    assert infer_hints["representation"] is RepresentationOutput
    assert helper_hints["representation"] is RepresentationOutput


def test_reference_basic_fit_and_infer_accept_representation_output() -> None:
    """ReferenceBasicNormality should train on typed pixel outputs and emit normalized artifacts."""

    generator = torch.Generator().manual_seed(0)
    samples = [
        Sample(
            image=torch.rand(3, 16, 16, generator=generator),
            reference=torch.rand(3, 16, 16, generator=generator),
            sample_id=f"sample-{index + 1:03d}",
        )
        for index in range(2)
    ]
    representations = [make_pixel_output(sample.image, sample_id=sample.sample_id or "sample") for sample in samples]
    model = ReferenceBasicNormality(
        input_channels=3,
        hidden_channels=8,
        learning_rate=1e-3,
        epochs=1,
        batch_size=2,
    )

    model.fit(representations, samples)
    artifacts = model.infer(samples[0], representations[0])

    assert artifacts.has("reference_projection")
    assert artifacts.has("conditional_alignment")
    assert "anomaly_map" not in artifacts.primary
    assert "image_score" not in artifacts.primary
    reference_projection = artifacts.get_primary("reference_projection")
    conditional_alignment = artifacts.get_aux("conditional_alignment")
    assert isinstance(reference_projection, torch.Tensor)
    assert isinstance(conditional_alignment, torch.Tensor)
    assert reference_projection.shape == samples[0].image.shape
    assert conditional_alignment.shape == samples[0].image.shape[-2:]
    assert artifacts.representation == representations[0].to_artifact_dict()
    assert artifacts.representation["space"] == "pixel"
    assert artifacts.representation["provenance"]["representation_name"] == "pixel"
    assert ReferenceBasicNormality.accepted_spaces == frozenset({"pixel"})
    assert ReferenceBasicNormality.accepted_tensor_ranks == frozenset({3})


def test_reference_basic_accepts_pil_reference_fallback() -> None:
    """ReferenceBasicNormality should keep working with raw PIL references."""

    torch.manual_seed(0)
    sample = Sample(
        image=torch.rand(3, 16, 16),
        reference=Image.new("RGB", (12, 10), color=(0, 255, 0)),
        sample_id="sample-pil",
    )
    representation = make_pixel_output(sample.image, sample_id="sample-pil")
    model = ReferenceBasicNormality(
        input_channels=3,
        hidden_channels=8,
        learning_rate=1e-3,
        epochs=1,
        batch_size=1,
    )

    model.fit([representation], [sample])
    artifacts = model.infer(sample, representation)

    assert artifacts.get_primary("reference_projection").shape == (3, 16, 16)
    assert artifacts.representation["sample_id"] == "sample-pil"
