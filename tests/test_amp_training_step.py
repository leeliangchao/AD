"""Tests for AMP-aware trainable model runtime setup."""

from unittest.mock import patch

import torch

from adrf.normality.autoencoder import AutoEncoderNormality
from adrf.utils.runtime import configure_trainable_runtime


def _pixel_representation(image: torch.Tensor) -> dict[str, object]:
    return {
        "representation": image,
        "space_type": "pixel",
        "spatial_shape": tuple(image.shape[-2:]),
    }


def test_amp_runtime_setup_does_not_break_cpu_training() -> None:
    """Enabling AMP in config should still train safely on CPU fallback."""

    model = AutoEncoderNormality(input_channels=3, hidden_channels=4, latent_channels=8, epochs=1, batch_size=2)
    with patch("torch.cuda.is_available", return_value=False):
        configure_trainable_runtime(model, device=torch.device("cpu"), amp_enabled=True)

    train_representations = [
        _pixel_representation(torch.rand(3, 16, 16)),
        _pixel_representation(torch.rand(3, 16, 16)),
    ]
    model.fit(train_representations)

    assert model.amp_enabled is False
    assert model.runtime_device.type == "cpu"
    assert model.last_fit_loss is not None

