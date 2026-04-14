"""Tests for the minimal reference-conditioned normality model."""

from PIL import Image
import torch

from adrf.core.sample import Sample
from adrf.normality.reference_basic import ReferenceBasicNormality


def _pixel_representation(image: torch.Tensor) -> dict[str, object]:
    return {
        "representation": image,
        "space_type": "pixel",
        "spatial_shape": tuple(image.shape[-2:]),
    }


def test_reference_basic_fit_and_infer_emit_conditional_artifacts() -> None:
    """ReferenceBasicNormality should emit a projection and alignment artifacts."""

    torch.manual_seed(0)
    samples = [
        Sample(image=torch.rand(3, 16, 16), reference=torch.rand(3, 16, 16), sample_id="sample-001"),
        Sample(image=torch.rand(3, 16, 16), reference=torch.rand(3, 16, 16), sample_id="sample-002"),
    ]
    representations = [_pixel_representation(sample.image) for sample in samples]
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


def test_reference_basic_accepts_pil_reference_fallback() -> None:
    """ReferenceBasicNormality should keep working with raw PIL references."""

    torch.manual_seed(0)
    sample = Sample(
        image=torch.rand(3, 16, 16),
        reference=Image.new("RGB", (12, 10), color=(0, 255, 0)),
        sample_id="sample-pil",
    )
    representation = _pixel_representation(sample.image)
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
