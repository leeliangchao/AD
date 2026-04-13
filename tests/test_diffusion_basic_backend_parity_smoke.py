"""Smoke tests for legacy and diffusers diffusion backends."""

import torch

from adrf.core.sample import Sample
from adrf.normality.diffusion_basic import DiffusionBasicNormality


def _pixel_representation(image: torch.Tensor) -> dict[str, object]:
    return {
        "representation": image,
        "space_type": "pixel",
        "spatial_shape": tuple(image.shape[-2:]),
    }


def test_diffusion_basic_legacy_and_diffusers_backends_share_artifact_contract() -> None:
    """Both backends should emit the same artifact keys and tensor shapes."""

    torch.manual_seed(0)
    train_representations = [
        _pixel_representation(torch.rand(3, 16, 16)),
        _pixel_representation(torch.rand(3, 16, 16)),
    ]
    sample = Sample(image=train_representations[0]["representation"], sample_id="sample-001")

    legacy = DiffusionBasicNormality(
        input_channels=3,
        hidden_channels=8,
        learning_rate=1e-3,
        epochs=1,
        batch_size=2,
        noise_level=0.2,
        backend="legacy",
    )
    legacy.fit(train_representations)
    legacy_artifacts = legacy.infer(sample, train_representations[0])

    diffusers = DiffusionBasicNormality(
        input_channels=3,
        hidden_channels=8,
        learning_rate=1e-3,
        epochs=1,
        batch_size=2,
        noise_level=0.2,
        backend="diffusers",
    )
    diffusers.fit(train_representations)
    diffusers_artifacts = diffusers.infer(sample, train_representations[0])

    for artifacts in (legacy_artifacts, diffusers_artifacts):
        assert artifacts.has("predicted_noise")
        assert artifacts.has("target_noise")

    assert legacy_artifacts.get_aux("predicted_noise").shape == diffusers_artifacts.get_aux("predicted_noise").shape
    assert legacy_artifacts.get_aux("target_noise").shape == diffusers_artifacts.get_aux("target_noise").shape
