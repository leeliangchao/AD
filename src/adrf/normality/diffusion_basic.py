"""Minimal diffusion-style normality model based on single-step noise prediction."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
import math
from typing import Sequence
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample
from adrf.diffusion.adapters import DiffusersNoisePredictorAdapter
from adrf.normality.base import BaseNormalityModel
from adrf.representation.contracts import RepresentationOutput


def _normalize_channel_mults(
    channel_mults: Sequence[int] | None,
    *,
    default: tuple[int, ...] = (1,),
) -> tuple[int, ...]:
    """Normalize a channel multiplier sequence into a validated tuple."""

    if channel_mults is None:
        return default
    normalized = tuple(int(multiplier) for multiplier in channel_mults)
    if not normalized or any(multiplier < 1 for multiplier in normalized):
        raise ValueError("channel_mults must be a non-empty sequence of positive integers.")
    return normalized


class _ResidualConvBlock(nn.Module):
    """A small residual conv block used to scale baseline denoiser capacity."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.act_out = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run one residual conv block."""

        residual = self.skip(inputs)
        hidden = self.act1(self.conv1(inputs))
        hidden = self.conv2(hidden)
        return self.act_out(hidden + residual)


def _sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    """Create sinusoidal timestep embeddings for diffusion conditioning."""

    if embedding_dim < 1:
        raise ValueError("embedding_dim must be at least 1.")

    half_dim = embedding_dim // 2
    if half_dim == 0:
        return timesteps.float().unsqueeze(1)

    exponent = -math.log(max_period) * torch.arange(
        half_dim,
        dtype=torch.float32,
        device=timesteps.device,
    ) / max(half_dim - 1, 1)
    frequencies = torch.exp(exponent)
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if embedding_dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
    return embedding


class _ConditionedResidualConvBlock(nn.Module):
    """Residual conv block with scale-shift conditioning."""

    def __init__(self, in_channels: int, out_channels: int, conditioning_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.condition_projection = nn.Linear(conditioning_dim, out_channels * 2)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.act_out = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Run one scale-shift-conditioned residual conv block."""

        residual = self.skip(inputs)
        hidden = self.conv1(inputs)
        scale, shift = self.condition_projection(conditioning).chunk(2, dim=1)
        hidden = hidden * (1.0 + torch.tanh(scale).unsqueeze(-1).unsqueeze(-1))
        hidden = hidden + shift.unsqueeze(-1).unsqueeze(-1)
        hidden = self.act1(hidden)
        hidden = self.conv2(hidden)
        return self.act_out(hidden + residual)


class _NoisePredictor(nn.Module):
    """Configurable fully-convolutional noise predictor for legacy diffusion baselines."""

    def __init__(
        self,
        input_channels: int,
        base_channels: int,
        channel_mults: Sequence[int],
        num_res_blocks: int,
        time_embed_dim: int,
        conditioning_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.time_embed_dim = int(time_embed_dim)
        self.conditioning_hidden_dim = int(conditioning_hidden_dim)
        self.condition_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim + 1, self.conditioning_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.conditioning_hidden_dim, self.time_embed_dim),
        )
        self.layers = nn.ModuleList()
        current_channels = input_channels
        for multiplier in channel_mults:
            stage_channels = base_channels * int(multiplier)
            self.layers.append(_ConditionedResidualConvBlock(current_channels, stage_channels, self.time_embed_dim))
            current_channels = stage_channels
            for _ in range(num_res_blocks - 1):
                self.layers.append(_ConditionedResidualConvBlock(current_channels, current_channels, self.time_embed_dim))
        self.output_projection = nn.Conv2d(current_channels, input_channels, kernel_size=3, padding=1)

    def forward(
        self,
        inputs: torch.Tensor,
        timesteps: torch.Tensor,
        noise_scales: torch.Tensor,
    ) -> torch.Tensor:
        """Predict per-pixel diffusion noise."""

        if timesteps.ndim != 1 or timesteps.shape[0] != inputs.shape[0]:
            raise ValueError("timesteps must be a 1D tensor aligned with the batch dimension.")
        if noise_scales.ndim != 1 or noise_scales.shape[0] != inputs.shape[0]:
            raise ValueError("noise_scales must be a 1D tensor aligned with the batch dimension.")
        time_embedding = _sinusoidal_timestep_embedding(timesteps, self.time_embed_dim)
        conditioning = torch.cat(
            [
                time_embedding,
                torch.log1p(noise_scales.float()).unsqueeze(1),
            ],
            dim=1,
        )
        conditioning = self.condition_mlp(conditioning).to(dtype=inputs.dtype)
        hidden = inputs
        for layer in self.layers:
            hidden = layer(hidden, conditioning)
        return self.output_projection(hidden)


class DiffusionBasicNormality(nn.Module, BaseNormalityModel):
    """Train a small denoiser on normal pixel samples and expose noise artifacts."""

    accepted_spaces = frozenset({"pixel"})
    accepted_tensor_ranks = frozenset({3})
    requires_detached_representation = True

    def __init__(
        self,
        input_channels: int = 3,
        hidden_channels: int = 16,
        base_channels: int | None = None,
        channel_mults: Sequence[int] | None = None,
        num_res_blocks: int = 2,
        time_embed_dim: int = 64,
        conditioning_hidden_dim: int | None = None,
        num_train_timesteps: int = 100,
        learning_rate: float = 1e-3,
        epochs: int = 1,
        batch_size: int = 8,
        noise_level: float = 0.2,
        backend: str = "legacy",
    ) -> None:
        super().__init__()
        resolved_base_channels = int(base_channels if base_channels is not None else hidden_channels)
        if resolved_base_channels < 1:
            raise ValueError("base_channels must be at least 1.")
        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be at least 1.")
        if time_embed_dim < 1:
            raise ValueError("time_embed_dim must be at least 1.")
        resolved_conditioning_hidden_dim = int(
            conditioning_hidden_dim if conditioning_hidden_dim is not None else max(time_embed_dim, resolved_base_channels * 2)
        )
        if resolved_conditioning_hidden_dim < 1:
            raise ValueError("conditioning_hidden_dim must be at least 1.")
        if num_train_timesteps < 2:
            raise ValueError("num_train_timesteps must be at least 2.")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.noise_level = noise_level
        self.backend = backend
        self.base_channels = resolved_base_channels
        self.hidden_channels = resolved_base_channels
        self.channel_mults = _normalize_channel_mults(channel_mults)
        self.num_res_blocks = int(num_res_blocks)
        self.time_embed_dim = int(time_embed_dim)
        self.conditioning_hidden_dim = resolved_conditioning_hidden_dim
        self.num_train_timesteps = int(num_train_timesteps)
        self.denoiser = _NoisePredictor(
            input_channels=input_channels,
            base_channels=self.base_channels,
            channel_mults=self.channel_mults,
            num_res_blocks=self.num_res_blocks,
            time_embed_dim=self.time_embed_dim,
            conditioning_hidden_dim=self.conditioning_hidden_dim,
        )
        self.diffusers_adapter: DiffusersNoisePredictorAdapter | None = None
        self.input_channels = input_channels
        self.last_fit_loss: float | None = None
        self.runtime_device = torch.device("cpu")
        self.amp_enabled = False
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=False)
        self._adrf_runtime_wrapped = True
        self.eval()

    def fit(
        self,
        representations: Iterable[RepresentationOutput],
        samples: Iterable[Sample] | None = None,
    ) -> None:
        """Train the denoiser on normal pixel-space representations."""

        del samples
        tensors = [self.require_representation_tensor(representation).float() for representation in representations]
        if not tensors:
            raise ValueError("DiffusionBasicNormality.fit requires at least one representation.")

        train_batch = torch.stack(tensors, dim=0)
        if self.backend == "diffusers":
            self._ensure_diffusers_backend(sample_size=int(train_batch.shape[-1]))
            optimizer = torch.optim.Adam(self.diffusers_adapter.parameters(), lr=self.learning_rate)
            self.diffusers_adapter.train()
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.train()
        for _ in range(self.epochs):
            permutation = torch.randperm(train_batch.shape[0])
            shuffled = train_batch[permutation]
            for start in range(0, shuffled.shape[0], self.batch_size):
                clean_batch = shuffled[start : start + self.batch_size]
                autocast_context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if self.amp_enabled and self.runtime_device.type == "cuda"
                    else nullcontext()
                )
                with autocast_context:
                    if self.backend == "diffusers":
                        loss, _ = self.diffusers_adapter.forward_train_step(clean_batch)
                    else:
                        noisy_batch, target_noise, timesteps, noise_scales = self._sample_noisy_inputs(clean_batch)
                        predicted_noise = self.denoiser(noisy_batch, timesteps, noise_scales)
                        loss = functional.mse_loss(predicted_noise, target_noise)
                optimizer.zero_grad()
                if self.grad_scaler.is_enabled():
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.step(optimizer)
                    self.grad_scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                self.last_fit_loss = float(loss.detach().cpu().item())
        self.eval()

    def infer(
        self,
        sample: Sample,
        representation: RepresentationOutput,
    ) -> NormalityArtifacts:
        """Run one single-step noise-prediction pass and emit diffusion artifacts."""

        clean_image = self.require_representation_tensor(representation).float().unsqueeze(0)
        with torch.no_grad():
            if self.backend == "diffusers":
                self._ensure_diffusers_backend(sample_size=int(clean_image.shape[-1]))
                predicted_noise, target_noise = self.diffusers_adapter.forward_infer_step(clean_image)
                inference_timestep = self.num_train_timesteps - 1
            else:
                noisy_image, target_noise, timesteps, noise_scales = self._sample_noisy_inputs(clean_image, inference=True)
                predicted_noise = self.denoiser(noisy_image, timesteps, noise_scales)
                inference_timestep = int(timesteps[0].item())
        return NormalityArtifacts(
            context={
                "sample_id": sample.sample_id,
                "category": sample.category,
                "mode": "inference",
            },
            representation=self.serialize_representation(representation),
            primary={},
            auxiliary={
                "predicted_noise": predicted_noise.squeeze(0),
                "target_noise": target_noise.squeeze(0),
            },
            diagnostics={
                "fit_loss": self.last_fit_loss,
                "noise_level": self.noise_level,
                "time_embed_dim": self.time_embed_dim,
                "conditioning_hidden_dim": self.conditioning_hidden_dim,
                "num_train_timesteps": self.num_train_timesteps,
                "inference_timestep": inference_timestep,
            },
            capabilities={"predicted_noise", "target_noise"},
        )

    def _sample_noisy_inputs(
        self,
        clean_batch: torch.Tensor,
        *,
        inference: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample Gaussian noise and produce the corresponding noisy inputs."""

        timesteps = self._sample_timesteps(clean_batch.shape[0], clean_batch.device, inference=inference)
        noise_scale = self._noise_scale_from_timesteps(timesteps).view(-1, 1, 1, 1)
        target_noise = torch.randn_like(clean_batch)
        noisy_batch = clean_batch + noise_scale * target_noise
        return noisy_batch, target_noise, timesteps, noise_scale.view(-1)

    def _sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        *,
        inference: bool = False,
    ) -> torch.Tensor:
        """Sample training timesteps or choose a deterministic inference timestep."""

        if inference:
            return torch.full(
                (batch_size,),
                fill_value=self.num_train_timesteps - 1,
                dtype=torch.long,
                device=device,
            )
        return torch.randint(
            low=0,
            high=self.num_train_timesteps,
            size=(batch_size,),
            dtype=torch.long,
            device=device,
        )

    def _noise_scale_from_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Map discrete timesteps onto bounded per-sample noise scales."""

        timestep_fraction = (timesteps.float() + 1.0) / float(self.num_train_timesteps)
        return self.noise_level * torch.sqrt(timestep_fraction)

    def _ensure_diffusers_backend(self, sample_size: int) -> None:
        """Instantiate the diffusers backend lazily when first needed."""

        if self.backend != "diffusers" or self.diffusers_adapter is not None:
            return
        self.diffusers_adapter = DiffusersNoisePredictorAdapter(
            input_channels=self.input_channels,
            hidden_channels=self.base_channels,
            learning_rate=self.learning_rate,
            noise_level=self.noise_level,
            sample_size=sample_size,
        )
        self.diffusers_adapter.to(self.runtime_device)
        self.diffusers_adapter.amp_enabled = self.amp_enabled
        self.diffusers_adapter.grad_scaler = self.grad_scaler
