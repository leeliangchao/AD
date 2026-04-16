"""Reusable diffusion model backbones and conditioning-aware denoisers."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from adrf.normality.diffusion_core import sinusoidal_timestep_embedding


class ResidualConvBlock(nn.Module):
    """A small residual conv block used to scale denoiser capacity."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_out = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.skip(inputs)
        hidden = self.act1(self.conv1(inputs))
        hidden = self.conv2(hidden)
        return self.act_out(hidden + residual)


class ConditionedResidualConvBlock(nn.Module):
    """Residual conv block with scale-shift conditioning."""

    def __init__(self, in_channels: int, out_channels: int, conditioning_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.condition_projection = nn.Linear(conditioning_dim, out_channels * 2)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_out = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        residual = self.skip(inputs)
        hidden = self.conv1(inputs)
        scale, shift = self.condition_projection(conditioning).chunk(2, dim=1)
        hidden = hidden * (1.0 + torch.tanh(scale).unsqueeze(-1).unsqueeze(-1))
        hidden = hidden + shift.unsqueeze(-1).unsqueeze(-1)
        hidden = self.act1(hidden)
        hidden = self.conv2(hidden)
        return self.act_out(hidden + residual)


class NoisePredictor(nn.Module):
    """Configurable fully-convolutional noise predictor with optional class conditioning."""

    def __init__(
        self,
        input_channels: int,
        base_channels: int,
        channel_mults: Sequence[int],
        num_res_blocks: int,
        time_embed_dim: int,
        conditioning_hidden_dim: int,
        *,
        num_classes: int | None = None,
        class_embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.time_embed_dim = int(time_embed_dim)
        class_condition_dim = 0
        if num_classes is not None:
            resolved_class_embed_dim = int(class_embed_dim if class_embed_dim is not None else self.time_embed_dim)
            self.class_embedding = nn.Embedding(int(num_classes), resolved_class_embed_dim)
            class_condition_dim = resolved_class_embed_dim
        else:
            self.class_embedding = None
        self.conditioning_hidden_dim = int(conditioning_hidden_dim)
        self.condition_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim + 1 + class_condition_dim, self.conditioning_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.conditioning_hidden_dim, self.time_embed_dim),
        )
        self.layers = nn.ModuleList()
        current_channels = input_channels
        for multiplier in channel_mults:
            stage_channels = base_channels * int(multiplier)
            self.layers.append(ConditionedResidualConvBlock(current_channels, stage_channels, self.time_embed_dim))
            current_channels = stage_channels
            for _ in range(num_res_blocks - 1):
                self.layers.append(ConditionedResidualConvBlock(current_channels, current_channels, self.time_embed_dim))
        self.output_projection = nn.Conv2d(current_channels, input_channels, kernel_size=3, padding=1)

    def forward(
        self,
        inputs: torch.Tensor,
        timesteps: torch.Tensor,
        noise_scales: torch.Tensor,
        *,
        class_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if timesteps.ndim != 1 or timesteps.shape[0] != inputs.shape[0]:
            raise ValueError("timesteps must be a 1D tensor aligned with the batch dimension.")
        if noise_scales.ndim != 1 or noise_scales.shape[0] != inputs.shape[0]:
            raise ValueError("noise_scales must be a 1D tensor aligned with the batch dimension.")
        time_embedding = sinusoidal_timestep_embedding(timesteps, self.time_embed_dim)
        conditioning_parts = [
            time_embedding,
            torch.log1p(noise_scales.float()).unsqueeze(1),
        ]
        if self.class_embedding is not None:
            if class_ids is None:
                raise ValueError("class_ids are required when class conditioning is enabled.")
            conditioning_parts.append(self.class_embedding(class_ids.to(device=inputs.device)))
        conditioning = torch.cat(conditioning_parts, dim=1)
        conditioning = self.condition_mlp(conditioning).to(dtype=inputs.dtype)
        hidden = inputs
        for layer in self.layers:
            hidden = layer(hidden, conditioning)
        return self.output_projection(hidden)


class ConditionedNoisePredictor(nn.Module):
    """Conditional denoiser with optional class conditioning."""

    def __init__(
        self,
        input_channels: int,
        base_channels: int,
        condition_channels: int,
        channel_mults: Sequence[int],
        num_res_blocks: int,
        time_embed_dim: int,
        *,
        num_classes: int | None = None,
        class_embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.time_embed_dim = int(time_embed_dim)
        hidden_embed_dim = max(self.time_embed_dim, base_channels * 2)
        class_condition_dim = 0
        if num_classes is not None:
            resolved_class_embed_dim = int(class_embed_dim if class_embed_dim is not None else self.time_embed_dim)
            self.class_embedding = nn.Embedding(int(num_classes), resolved_class_embed_dim)
            class_condition_dim = resolved_class_embed_dim
        else:
            self.class_embedding = None
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim + class_condition_dim, hidden_embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_embed_dim, self.time_embed_dim),
        )
        self.image_projection = nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1)
        self.condition_projection = nn.Conv2d(input_channels, condition_channels, kernel_size=3, padding=1)
        self.reference_injections = nn.ModuleList()
        self.layers = nn.ModuleList()
        current_channels = base_channels
        for multiplier in channel_mults:
            stage_channels = base_channels * int(multiplier)
            self.reference_injections.append(nn.Conv2d(condition_channels, current_channels, kernel_size=1))
            self.layers.append(ConditionedResidualConvBlock(current_channels, stage_channels, self.time_embed_dim))
            current_channels = stage_channels
            for _ in range(num_res_blocks - 1):
                self.reference_injections.append(nn.Conv2d(condition_channels, current_channels, kernel_size=1))
                self.layers.append(ConditionedResidualConvBlock(current_channels, current_channels, self.time_embed_dim))
        self.output_projection = nn.Conv2d(current_channels, input_channels, kernel_size=3, padding=1)

    def forward(
        self,
        inputs: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        class_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        image, reference = torch.split(inputs, [self.input_channels, inputs.shape[1] - self.input_channels], dim=1)
        hidden = self.image_projection(image)
        reference_features = self.condition_projection(reference)
        conditioning_parts = [sinusoidal_timestep_embedding(timesteps, self.time_embed_dim)]
        if self.class_embedding is not None:
            if class_ids is None:
                raise ValueError("class_ids are required when class conditioning is enabled.")
            conditioning_parts.append(self.class_embedding(class_ids.to(device=inputs.device)))
        time_embedding = torch.cat(conditioning_parts, dim=1)
        time_embedding = self.time_mlp(time_embedding).to(dtype=inputs.dtype)
        for reference_adapter, layer in zip(self.reference_injections, self.layers, strict=True):
            hidden = hidden + reference_adapter(reference_features)
            hidden = layer(hidden, time_embedding)
        return self.output_projection(hidden)
