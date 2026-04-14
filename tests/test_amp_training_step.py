"""Tests for AMP-aware trainable model runtime setup."""

from pathlib import Path
import sys
from unittest.mock import patch

import torch

from adrf.normality.autoencoder import AutoEncoderNormality
from adrf.utils.runtime import configure_trainable_runtime

sys.path.insert(0, str(Path(__file__).parent))

from support.representation_builders import make_pixel_output


def test_amp_runtime_setup_does_not_break_cpu_training() -> None:
    """Enabling AMP in config should still train safely on CPU fallback."""

    model = AutoEncoderNormality(input_channels=3, hidden_channels=4, latent_channels=8, epochs=1, batch_size=2)
    with patch("torch.cuda.is_available", return_value=False):
        configure_trainable_runtime(model, device=torch.device("cpu"), amp_enabled=True)

    generator = torch.Generator().manual_seed(0)
    train_representations = [
        make_pixel_output(torch.rand(3, 16, 16, generator=generator), sample_id=f"train-{index:03d}")
        for index in range(2)
    ]
    model.fit(train_representations)

    assert model.amp_enabled is False
    assert model.runtime_device.type == "cpu"
    assert model.last_fit_loss is not None
