"""Tests for the diffusers UNet adapter."""

import torch

from adrf.diffusion.models import make_unet_model


def test_diffusers_unet_adapter_forward_shape_matches_input() -> None:
    """The diffusers UNet adapter should emit a noise prediction with input shape."""

    model = make_unet_model(
        {
            "sample_size": 16,
            "in_channels": 3,
            "out_channels": 3,
            "layers_per_block": 1,
            "block_out_channels": (8, 16),
            "norm_num_groups": 4,
        }
    )
    sample = torch.rand(2, 3, 16, 16)
    timesteps = torch.tensor([1, 2], dtype=torch.long)

    prediction = model(sample, timesteps)

    assert prediction.shape == sample.shape

