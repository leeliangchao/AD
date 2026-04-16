"""Minimal diffusion-style normality model based on single-step noise prediction."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from typing import Sequence
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample
from adrf.diffusion.adapters import DiffusersNoisePredictorAdapter
from adrf.normality.base import BaseNormalityModel
from adrf.normality.diffusion_core import (
    deterministic_noise_like,
    legacy_noise_scale_from_timesteps,
    legacy_reconstruct_clean,
    normalize_channel_mults,
    sample_legacy_noisy_inputs,
    validate_diffusion_backend,
)
from adrf.normality.diffusion_conditioning import resolve_optional_class_ids
from adrf.normality.diffusion_models import ConditionedResidualConvBlock, NoisePredictor, ResidualConvBlock
from adrf.normality.diffusion_tasks import build_reconstruction_artifacts
from adrf.normality.state import install_normality_runtime_state, make_default_normality_runtime_state
from adrf.representation.contracts import RepresentationOutput

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
        num_classes: int | None = None,
        class_embed_dim: int | None = None,
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
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.class_embed_dim = int(class_embed_dim) if class_embed_dim is not None else None
        self.backend = validate_diffusion_backend(
            backend,
            supported_backends=("legacy", "diffusers"),
            model_name=type(self).__name__,
        )
        self.base_channels = resolved_base_channels
        self.hidden_channels = resolved_base_channels
        self.channel_mults = normalize_channel_mults(channel_mults)
        self.num_res_blocks = int(num_res_blocks)
        self.time_embed_dim = int(time_embed_dim)
        self.conditioning_hidden_dim = resolved_conditioning_hidden_dim
        self.num_train_timesteps = int(num_train_timesteps)
        self.denoiser = NoisePredictor(
            input_channels=input_channels,
            base_channels=self.base_channels,
            channel_mults=self.channel_mults,
            num_res_blocks=self.num_res_blocks,
            time_embed_dim=self.time_embed_dim,
            conditioning_hidden_dim=self.conditioning_hidden_dim,
            num_classes=self.num_classes,
            class_embed_dim=self.class_embed_dim,
        )
        self.diffusers_adapter: DiffusersNoisePredictorAdapter | None = None
        self.input_channels = input_channels
        self.last_fit_loss: float | None = None
        self.class_to_index: dict[str, int] = {}
        install_normality_runtime_state(self, make_default_normality_runtime_state())
        self._adrf_runtime_wrapped = True
        self.eval()

    def fit(
        self,
        representations: Iterable[RepresentationOutput],
        samples: Iterable[Sample] | None = None,
    ) -> None:
        """Train the denoiser on normal pixel-space representations."""

        sample_list = list(samples) if samples is not None else None
        tensors = [self.require_representation_tensor(representation).float() for representation in representations]
        if not tensors:
            raise ValueError("DiffusionBasicNormality.fit requires at least one representation.")

        train_batch = torch.stack(tensors, dim=0)
        train_class_ids = resolve_optional_class_ids(
            sample_list,
            num_classes=self.num_classes,
            class_to_index=self.class_to_index,
            fit=True,
            backend=self.backend,
            supported_backends=("legacy", "diffusers"),
            model_name=type(self).__name__,
        )
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
            shuffled_class_ids = train_class_ids[permutation] if train_class_ids is not None else None
            for start in range(0, shuffled.shape[0], self.batch_size):
                clean_batch = shuffled[start : start + self.batch_size]
                batch_class_ids = (
                    shuffled_class_ids[start : start + self.batch_size].to(device=clean_batch.device)
                    if shuffled_class_ids is not None
                    else None
                )
                autocast_context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if self.runtime.amp_enabled and self.runtime.device.type == "cuda"
                    else nullcontext()
                )
                with autocast_context:
                    if self.backend == "diffusers":
                        loss, _ = self.diffusers_adapter.forward_train_step(clean_batch, class_ids=batch_class_ids)
                    else:
                        noisy_batch, target_noise, timesteps, noise_scales = self._sample_noisy_inputs(clean_batch)
                        predicted_noise = self.denoiser(
                            noisy_batch,
                            timesteps,
                            noise_scales,
                            class_ids=batch_class_ids,
                        )
                        loss = functional.mse_loss(predicted_noise, target_noise)
                optimizer.zero_grad()
                if self.runtime.grad_scaler.is_enabled():
                    self.runtime.grad_scaler.scale(loss).backward()
                    self.runtime.grad_scaler.step(optimizer)
                    self.runtime.grad_scaler.update()
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
        inference_identity = sample.sample_id or representation.sample_id or "inference"
        class_ids = resolve_optional_class_ids(
            [sample],
            num_classes=self.num_classes,
            class_to_index=self.class_to_index,
            fit=False,
            backend=self.backend,
            supported_backends=("legacy", "diffusers"),
            model_name=type(self).__name__,
        )
        if class_ids is not None:
            class_ids = class_ids.to(device=clean_image.device)
        target_noise = deterministic_noise_like(
            clean_image,
            type(self).__name__,
            inference_identity,
            self.num_train_timesteps,
            self.noise_level,
        )
        with torch.no_grad():
            if self.backend == "diffusers":
                self._ensure_diffusers_backend(sample_size=int(clean_image.shape[-1]))
                predicted_noise, target_noise, noisy_image, timesteps = self.diffusers_adapter.forward_infer_step(
                    clean_image,
                    target_noise=target_noise,
                    class_ids=class_ids,
                )
                reconstruction = self.diffusers_adapter.reconstruct_clean(noisy_image, predicted_noise, timesteps)
                inference_timestep = int(timesteps[0].item())
            else:
                noisy_image, target_noise, timesteps, noise_scales = self._sample_noisy_inputs(
                    clean_image,
                    inference=True,
                    target_noise=target_noise,
                )
                predicted_noise = self.denoiser(noisy_image, timesteps, noise_scales, class_ids=class_ids)
                reconstruction = legacy_reconstruct_clean(noisy_image, predicted_noise, noise_scales)
                inference_timestep = int(timesteps[0].item())
        return build_reconstruction_artifacts(
            sample_id=sample.sample_id,
            category=sample.category,
            representation=self.serialize_representation(representation),
            reconstruction=reconstruction.squeeze(0),
            predicted_noise=predicted_noise.squeeze(0),
            target_noise=target_noise.squeeze(0),
            diagnostics={
                "fit_loss": self.last_fit_loss,
                "noise_level": self.noise_level,
                "time_embed_dim": self.time_embed_dim,
                "conditioning_hidden_dim": self.conditioning_hidden_dim,
                "num_train_timesteps": self.num_train_timesteps,
                "inference_timestep": inference_timestep,
            },
        )

    def _sample_noisy_inputs(
        self,
        clean_batch: torch.Tensor,
        *,
        inference: bool = False,
        target_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample Gaussian noise and produce the corresponding noisy inputs."""

        return sample_legacy_noisy_inputs(
            clean_batch,
            num_train_timesteps=self.num_train_timesteps,
            noise_level=self.noise_level,
            inference=inference,
            target_noise=target_noise,
        )

    def _noise_scale_from_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Map discrete timesteps onto bounded per-sample noise scales."""

        return legacy_noise_scale_from_timesteps(
            timesteps,
            num_train_timesteps=self.num_train_timesteps,
            noise_level=self.noise_level,
        )

    def _ensure_diffusers_backend(self, sample_size: int) -> None:
        """Instantiate the diffusers backend lazily when first needed."""

        if self.backend != "diffusers" or self.diffusers_adapter is not None:
            return
        self.diffusers_adapter = DiffusersNoisePredictorAdapter(
            input_channels=self.input_channels,
            output_channels=None,
            hidden_channels=self.base_channels,
            learning_rate=self.learning_rate,
            noise_level=self.noise_level,
            sample_size=sample_size,
            num_train_timesteps=self.num_train_timesteps,
            num_classes=self.num_classes,
        )
        self.diffusers_adapter.to(self.runtime.device)
        install_normality_runtime_state(self.diffusers_adapter, self.runtime)
