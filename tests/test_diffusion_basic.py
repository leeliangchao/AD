"""Tests for the minimal diffusion-style normality model."""

import torch

from adrf.core.sample import Sample
from adrf.normality.diffusion_basic import DiffusionBasicNormality


def _pixel_representation(image: torch.Tensor) -> dict[str, object]:
    return {
        "representation": image,
        "space_type": "pixel",
        "spatial_shape": tuple(image.shape[-2:]),
    }


def test_diffusion_basic_fit_and_infer_emit_noise_artifacts() -> None:
    """DiffusionBasicNormality should train on normal pixels and emit noise artifacts."""

    torch.manual_seed(0)
    train_representations = [
        _pixel_representation(torch.rand(3, 16, 16)),
        _pixel_representation(torch.rand(3, 16, 16)),
    ]
    model = DiffusionBasicNormality(
        input_channels=3,
        hidden_channels=8,
        learning_rate=1e-3,
        epochs=1,
        batch_size=2,
        noise_level=0.2,
    )

    model.fit(train_representations)
    sample = Sample(image=train_representations[0]["representation"], sample_id="sample-001")
    artifacts = model.infer(sample, train_representations[0])

    assert artifacts.has("predicted_noise")
    assert artifacts.has("target_noise")
    assert "anomaly_map" not in artifacts.primary
    predicted_noise = artifacts.get_aux("predicted_noise")
    target_noise = artifacts.get_aux("target_noise")
    assert isinstance(predicted_noise, torch.Tensor)
    assert isinstance(target_noise, torch.Tensor)
    assert predicted_noise.shape == target_noise.shape == (3, 16, 16)

