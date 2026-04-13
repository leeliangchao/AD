"""Tests for the minimal reference-conditioned diffusion normality model."""

import torch

from adrf.core.sample import Sample
from adrf.normality.reference_diffusion_basic import ReferenceDiffusionBasicNormality


def _pixel_representation(image: torch.Tensor) -> dict[str, object]:
    return {
        "representation": image,
        "space_type": "pixel",
        "spatial_shape": tuple(image.shape[-2:]),
    }


def test_reference_diffusion_basic_fit_and_infer_emit_diffusion_artifacts() -> None:
    """ReferenceDiffusionBasicNormality should emit conditional diffusion artifacts."""

    torch.manual_seed(0)
    samples = [
        Sample(image=torch.rand(3, 16, 16), reference=torch.rand(3, 16, 16), sample_id="sample-001"),
        Sample(image=torch.rand(3, 16, 16), reference=torch.rand(3, 16, 16), sample_id="sample-002"),
    ]
    representations = [_pixel_representation(sample.image) for sample in samples]
    model = ReferenceDiffusionBasicNormality(
        input_channels=3,
        hidden_channels=8,
        learning_rate=1e-3,
        epochs=1,
        batch_size=2,
        noise_level=0.2,
    )

    model.fit(representations, samples)
    artifacts = model.infer(samples[0], representations[0])

    assert artifacts.has("predicted_noise")
    assert artifacts.has("target_noise")
    assert artifacts.has("reference_projection")
    assert artifacts.has("conditional_alignment")
    assert "anomaly_map" not in artifacts.primary
    assert "image_score" not in artifacts.primary

    predicted_noise = artifacts.get_aux("predicted_noise")
    target_noise = artifacts.get_aux("target_noise")
    reference_projection = artifacts.get_primary("reference_projection")
    conditional_alignment = artifacts.get_aux("conditional_alignment")
    assert isinstance(predicted_noise, torch.Tensor)
    assert isinstance(target_noise, torch.Tensor)
    assert isinstance(reference_projection, torch.Tensor)
    assert isinstance(conditional_alignment, torch.Tensor)
    assert predicted_noise.shape == target_noise.shape == (3, 16, 16)
    assert reference_projection.shape == (3, 16, 16)
    assert conditional_alignment.shape == (16, 16)

