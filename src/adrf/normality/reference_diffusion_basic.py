"""Minimal reference-conditioned diffusion normality model."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Sequence

import torch
import torch.nn.functional as functional
from PIL import Image
from torch import nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tv_functional

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample
from adrf.normality.diffusion_basic import (
    _ConditionedResidualConvBlock,
    _normalize_channel_mults,
    _sinusoidal_timestep_embedding,
)
from adrf.normality.base import BaseNormalityModel
from adrf.normality.state import install_normality_runtime_state, make_default_normality_runtime_state
from adrf.representation.contracts import RepresentationOutput


class _ConditionedDenoiser(nn.Module):
    """Configurable conditional denoiser with explicit image/reference fusion width."""

    def __init__(
        self,
        input_channels: int,
        base_channels: int,
        condition_channels: int,
        channel_mults: Sequence[int],
        num_res_blocks: int,
        time_embed_dim: int,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.time_embed_dim = int(time_embed_dim)
        hidden_embed_dim = max(self.time_embed_dim, base_channels * 2)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, hidden_embed_dim),
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
            self.layers.append(_ConditionedResidualConvBlock(current_channels, stage_channels, self.time_embed_dim))
            current_channels = stage_channels
            for _ in range(num_res_blocks - 1):
                self.reference_injections.append(nn.Conv2d(condition_channels, current_channels, kernel_size=1))
                self.layers.append(_ConditionedResidualConvBlock(current_channels, current_channels, self.time_embed_dim))
        self.output_projection = nn.Conv2d(current_channels, input_channels, kernel_size=3, padding=1)

    def forward(self, inputs: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict conditional noise from concatenated image/reference inputs."""

        image, reference = torch.split(inputs, [self.input_channels, inputs.shape[1] - self.input_channels], dim=1)
        hidden = self.image_projection(image)
        reference_features = self.condition_projection(reference)
        time_embedding = _sinusoidal_timestep_embedding(timesteps, self.time_embed_dim)
        time_embedding = self.time_mlp(time_embedding).to(dtype=inputs.dtype)
        for reference_adapter, layer in zip(self.reference_injections, self.layers, strict=True):
            hidden = hidden + reference_adapter(reference_features)
            hidden = layer(hidden, time_embedding)
        return self.output_projection(hidden)


class ReferenceDiffusionBasicNormality(nn.Module, BaseNormalityModel):
    """Predict diffusion noise under a fixed per-category reference condition."""

    accepted_spaces = frozenset({"pixel"})
    accepted_tensor_ranks = frozenset({3})
    requires_detached_representation = True

    def __init__(
        self,
        input_channels: int = 3,
        hidden_channels: int = 16,
        base_channels: int | None = None,
        condition_channels: int | None = None,
        channel_mults: Sequence[int] | None = None,
        num_res_blocks: int = 2,
        time_embed_dim: int = 64,
        num_train_timesteps: int = 100,
        learning_rate: float = 1e-3,
        epochs: int = 1,
        batch_size: int = 8,
        noise_level: float = 0.2,
    ) -> None:
        super().__init__()
        resolved_base_channels = int(base_channels if base_channels is not None else hidden_channels)
        if resolved_base_channels < 1:
            raise ValueError("base_channels must be at least 1.")
        resolved_condition_channels = int(
            condition_channels if condition_channels is not None else resolved_base_channels
        )
        if resolved_condition_channels < 1:
            raise ValueError("condition_channels must be at least 1.")
        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be at least 1.")
        if time_embed_dim < 1:
            raise ValueError("time_embed_dim must be at least 1.")
        if num_train_timesteps < 2:
            raise ValueError("num_train_timesteps must be at least 2.")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.noise_level = noise_level
        self.base_channels = resolved_base_channels
        self.hidden_channels = resolved_base_channels
        self.condition_channels = resolved_condition_channels
        self.channel_mults = _normalize_channel_mults(channel_mults)
        self.num_res_blocks = int(num_res_blocks)
        self.time_embed_dim = int(time_embed_dim)
        self.num_train_timesteps = int(num_train_timesteps)
        self.conditional_denoiser = _ConditionedDenoiser(
            input_channels=input_channels,
            base_channels=self.base_channels,
            condition_channels=self.condition_channels,
            channel_mults=self.channel_mults,
            num_res_blocks=self.num_res_blocks,
            time_embed_dim=self.time_embed_dim,
        )
        self.last_fit_loss: float | None = None
        install_normality_runtime_state(self, make_default_normality_runtime_state())
        self.eval()

    def fit(
        self,
        representations: Iterable[RepresentationOutput],
        samples: Iterable[Sample] | None = None,
    ) -> None:
        """Train a reference-conditioned denoiser on normal pixel samples."""

        if samples is None:
            raise ValueError("ReferenceDiffusionBasicNormality.fit requires samples with reference inputs.")

        paired_tensors = [
            (
                self.require_representation_tensor(representation).float(),
                self._prepare_reference_tensor(sample, representation).float(),
            )
            for representation, sample in zip(representations, samples, strict=True)
        ]
        if not paired_tensors:
            raise ValueError("ReferenceDiffusionBasicNormality.fit requires at least one representation.")

        image_batch = torch.stack([image for image, _ in paired_tensors], dim=0)
        reference_batch = torch.stack([reference for _, reference in paired_tensors], dim=0)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.train()
        for _ in range(self.epochs):
            permutation = torch.randperm(image_batch.shape[0])
            shuffled_images = image_batch[permutation]
            shuffled_references = reference_batch[permutation]
            for start in range(0, shuffled_images.shape[0], self.batch_size):
                clean_batch = shuffled_images[start : start + self.batch_size]
                reference_slice = shuffled_references[start : start + self.batch_size]
                noisy_batch, target_noise, timesteps = self._sample_noisy_inputs(clean_batch)
                conditional_inputs = torch.cat([noisy_batch, reference_slice], dim=1)
                predicted_noise = self.conditional_denoiser(conditional_inputs, timesteps)
                loss = functional.mse_loss(predicted_noise, target_noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.last_fit_loss = float(loss.detach().cpu().item())
        self.eval()

    def infer(
        self,
        sample: Sample,
        representation: RepresentationOutput,
    ) -> NormalityArtifacts:
        """Infer conditional diffusion artifacts for one sample."""

        clean_image = self.require_representation_tensor(representation).float().unsqueeze(0)
        reference = self._prepare_reference_tensor(sample, representation).float().unsqueeze(0)
        with torch.no_grad():
            noisy_image, target_noise, timesteps = self._sample_noisy_inputs(clean_image, inference=True)
            conditional_inputs = torch.cat([noisy_image, reference], dim=1)
            predicted_noise = self.conditional_denoiser(conditional_inputs, timesteps)
            noise_scale = self._noise_scale_from_timesteps(timesteps).view(-1, 1, 1, 1)
            reference_projection = noisy_image - noise_scale * predicted_noise
        conditional_alignment = torch.abs(reference_projection - reference).mean(dim=1).squeeze(0)

        return NormalityArtifacts(
            context={
                "sample_id": sample.sample_id,
                "category": sample.category,
                "has_reference": sample.has_reference(),
                "mode": "inference",
            },
            representation=self.serialize_representation(representation),
            primary={"reference_projection": reference_projection.squeeze(0)},
            auxiliary={
                "predicted_noise": predicted_noise.squeeze(0),
                "target_noise": target_noise.squeeze(0),
                "conditional_alignment": conditional_alignment,
            },
            diagnostics={
                "fit_loss": self.last_fit_loss,
                "noise_level": self.noise_level,
                "time_embed_dim": self.time_embed_dim,
                "num_train_timesteps": self.num_train_timesteps,
                "inference_timestep": int(timesteps[0].item()),
            },
            capabilities={
                "predicted_noise",
                "target_noise",
                "reference_projection",
                "conditional_alignment",
            },
        )

    def _sample_noisy_inputs(
        self,
        clean_batch: torch.Tensor,
        *,
        inference: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample Gaussian noise and produce the corresponding noisy inputs."""

        timesteps = self._sample_timesteps(clean_batch.shape[0], clean_batch.device, inference=inference)
        noise_scale = self._noise_scale_from_timesteps(timesteps).view(-1, 1, 1, 1)
        target_noise = torch.randn_like(clean_batch)
        noisy_batch = clean_batch + noise_scale * target_noise
        return noisy_batch, target_noise, timesteps

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

    def _prepare_reference_tensor(
        self,
        sample: Sample,
        representation: RepresentationOutput,
    ) -> torch.Tensor:
        """Convert the sample reference into a tensor aligned with the image representation."""

        if sample.reference is None:
            raise ValueError("ReferenceDiffusionBasicNormality requires sample.reference to be present.")

        image = self.require_representation_tensor(representation)
        target_size = tuple(image.shape[-2:])
        reference = sample.reference
        if isinstance(reference, torch.Tensor):
            reference_tensor = reference.float()
            if reference_tensor.ndim == 4 and reference_tensor.shape[0] == 1:
                reference_tensor = reference_tensor.squeeze(0)
            if reference_tensor.ndim != 3:
                raise TypeError("sample.reference tensor must have shape [C, H, W].")
            if tuple(reference_tensor.shape[-2:]) != target_size:
                reference_tensor = tv_functional.resize(
                    reference_tensor,
                    target_size,
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )
            return reference_tensor.to(dtype=image.dtype, device=image.device)

        if isinstance(reference, Image.Image):
            resized = tv_functional.resize(
                reference,
                target_size,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            reference_tensor = tv_functional.to_tensor(resized)
            return reference_tensor.to(dtype=image.dtype, device=image.device)

        raise TypeError(
            "ReferenceDiffusionBasicNormality expects sample.reference to be a PIL image or torch.Tensor."
        )
