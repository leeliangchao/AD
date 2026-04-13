"""Tests for DiffusionInversionBasicNormality with the diffusers backend."""

import torch

from adrf.core.sample import Sample
from adrf.normality.diffusion_inversion_basic import DiffusionInversionBasicNormality


def _pixel_representation(image: torch.Tensor) -> dict[str, object]:
    return {
        "representation": image,
        "space_type": "pixel",
        "spatial_shape": tuple(image.shape[-2:]),
    }


def test_diffusion_inversion_diffusers_backend_fit_and_infer_emit_process_artifacts() -> None:
    """The diffusers backend should expose the same process artifact contract."""

    torch.manual_seed(0)
    train_representations = [
        _pixel_representation(torch.rand(3, 16, 16)),
        _pixel_representation(torch.rand(3, 16, 16)),
    ]
    model = DiffusionInversionBasicNormality(
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

    model.fit(train_representations)
    sample = Sample(image=train_representations[0]["representation"], sample_id="sample-001")
    artifacts = model.infer(sample, train_representations[0])

    assert artifacts.has("trajectory")
    assert artifacts.has("step_costs")
    assert isinstance(artifacts.get_aux("trajectory"), list)
    assert isinstance(artifacts.get_aux("step_costs"), list)

