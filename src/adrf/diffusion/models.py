"""Model adapters built on top of diffusers UNet components."""

from __future__ import annotations

from typing import Any

from diffusers import UNet2DModel
import torch
from torch import nn


class DiffusersUNetAdapter(nn.Module):
    """Small wrapper around diffusers UNet2DModel with tensor-in/tensor-out API."""

    def __init__(self, model: UNet2DModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, sample: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Run the wrapped UNet and return only the predicted tensor."""

        return self.model(sample, timesteps).sample


def make_unet_model(config: dict[str, Any]) -> DiffusersUNetAdapter:
    """Construct a minimal diffusers UNet adapter from config."""

    model = UNet2DModel(
        sample_size=config.get("sample_size", 16),
        in_channels=int(config.get("in_channels", 3)),
        out_channels=int(config.get("out_channels", 3)),
        layers_per_block=int(config.get("layers_per_block", 1)),
        block_out_channels=tuple(config.get("block_out_channels", (8, 16))),
        down_block_types=tuple(config.get("down_block_types", ("DownBlock2D", "DownBlock2D"))),
        up_block_types=tuple(config.get("up_block_types", ("UpBlock2D", "UpBlock2D"))),
        norm_num_groups=int(config.get("norm_num_groups", 4)),
    )
    return DiffusersUNetAdapter(model)
