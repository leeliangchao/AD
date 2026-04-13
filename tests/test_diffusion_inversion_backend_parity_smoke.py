"""Parity smoke tests for legacy and diffusers inversion backends."""

import torch

from adrf.core.sample import Sample
from adrf.normality.diffusion_inversion_basic import DiffusionInversionBasicNormality


def _pixel_representation(image: torch.Tensor) -> dict[str, object]:
    return {
        "representation": image,
        "space_type": "pixel",
        "spatial_shape": tuple(image.shape[-2:]),
    }


def test_diffusion_inversion_legacy_and_diffusers_share_process_artifact_shapes() -> None:
    """Both inversion backends should emit the same process artifact shapes."""

    torch.manual_seed(0)
    train_representations = [
        _pixel_representation(torch.rand(3, 16, 16)),
        _pixel_representation(torch.rand(3, 16, 16)),
    ]
    sample = Sample(image=train_representations[0]["representation"], sample_id="sample-001")

    legacy = DiffusionInversionBasicNormality(
        input_channels=3,
        hidden_channels=8,
        learning_rate=1e-3,
        epochs=1,
        batch_size=2,
        noise_level=0.2,
        num_steps=4,
        step_size=0.1,
        backend="legacy",
    )
    legacy.fit(train_representations)
    legacy_artifacts = legacy.infer(sample, train_representations[0])

    diffusers = DiffusionInversionBasicNormality(
        input_channels=3,
        hidden_channels=8,
        learning_rate=1e-3,
        epochs=1,
        batch_size=2,
        noise_level=0.2,
        num_steps=4,
        step_size=0.1,
        backend="diffusers",
    )
    diffusers.fit(train_representations)
    diffusers_artifacts = diffusers.infer(sample, train_representations[0])

    assert len(legacy_artifacts.get_aux("trajectory")) == len(diffusers_artifacts.get_aux("trajectory"))
    assert len(legacy_artifacts.get_aux("step_costs")) == len(diffusers_artifacts.get_aux("step_costs"))

