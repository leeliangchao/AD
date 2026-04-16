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
from adrf.diffusion.adapters import DiffusersNoisePredictorAdapter
from adrf.normality.diffusion_core import (
    deterministic_noise_like,
    legacy_reconstruct_clean,
    normalize_channel_mults,
    sample_legacy_noisy_inputs,
    validate_diffusion_backend,
)
from adrf.normality.diffusion_conditioning import resolve_optional_class_ids
from adrf.normality.diffusion_models import ConditionedNoisePredictor
from adrf.normality.diffusion_tasks import build_conditional_reconstruction_artifacts
from adrf.normality.base import BaseNormalityModel
from adrf.normality.state import install_normality_runtime_state, make_default_normality_runtime_state
from adrf.representation.contracts import RepresentationOutput


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
        num_classes: int | None = None,
        class_embed_dim: int | None = None,
        backend: str = "legacy",
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
        self.backend = validate_diffusion_backend(
            backend,
            supported_backends=("legacy", "diffusers"),
            model_name=type(self).__name__,
        )
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.class_embed_dim = int(class_embed_dim) if class_embed_dim is not None else None
        self.input_channels = int(input_channels)
        self.base_channels = resolved_base_channels
        self.hidden_channels = resolved_base_channels
        self.condition_channels = resolved_condition_channels
        self.channel_mults = normalize_channel_mults(channel_mults)
        self.num_res_blocks = int(num_res_blocks)
        self.time_embed_dim = int(time_embed_dim)
        self.num_train_timesteps = int(num_train_timesteps)
        self.conditional_denoiser = ConditionedNoisePredictor(
            input_channels=input_channels,
            base_channels=self.base_channels,
            condition_channels=self.condition_channels,
            channel_mults=self.channel_mults,
            num_res_blocks=self.num_res_blocks,
            time_embed_dim=self.time_embed_dim,
            num_classes=self.num_classes,
            class_embed_dim=self.class_embed_dim,
        )
        self.diffusers_adapter: DiffusersNoisePredictorAdapter | None = None
        self.last_fit_loss: float | None = None
        self.class_to_index: dict[str, int] = {}
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
        sample_list = list(samples)
        class_ids = resolve_optional_class_ids(
            sample_list,
            num_classes=self.num_classes,
            class_to_index=self.class_to_index,
            fit=True,
            backend=self.backend,
            supported_backends=("legacy", "diffusers"),
            model_name=type(self).__name__,
        )

        paired_tensors = [
            (
                self.require_representation_tensor(representation).float(),
                self._prepare_reference_tensor(sample, representation).float(),
            )
            for representation, sample in zip(representations, sample_list, strict=True)
        ]
        if not paired_tensors:
            raise ValueError("ReferenceDiffusionBasicNormality.fit requires at least one representation.")

        image_batch = torch.stack([image for image, _ in paired_tensors], dim=0)
        reference_batch = torch.stack([reference for _, reference in paired_tensors], dim=0)

        if self.backend == "diffusers":
            self._ensure_diffusers_backend(sample_size=int(image_batch.shape[-1]))
            optimizer = torch.optim.Adam(self.diffusers_adapter.parameters(), lr=self.learning_rate)
            self.diffusers_adapter.train()
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.train()
        for _ in range(self.epochs):
            permutation = torch.randperm(image_batch.shape[0])
            shuffled_images = image_batch[permutation]
            shuffled_references = reference_batch[permutation]
            shuffled_class_ids = class_ids[permutation] if class_ids is not None else None
            for start in range(0, shuffled_images.shape[0], self.batch_size):
                clean_batch = shuffled_images[start : start + self.batch_size]
                reference_slice = shuffled_references[start : start + self.batch_size]
                batch_class_ids = (
                    shuffled_class_ids[start : start + self.batch_size].to(device=clean_batch.device)
                    if shuffled_class_ids is not None
                    else None
                )
                if self.backend == "diffusers":
                    loss, _ = self.diffusers_adapter.forward_train_step(
                        clean_batch,
                        conditioning=reference_slice,
                        class_ids=batch_class_ids,
                    )
                else:
                    noisy_batch, target_noise, timesteps = self._sample_noisy_inputs(clean_batch)
                    conditional_inputs = torch.cat([noisy_batch, reference_slice], dim=1)
                    predicted_noise = self.conditional_denoiser(conditional_inputs, timesteps, class_ids=batch_class_ids)
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
        inference_identity = sample.sample_id or representation.sample_id or "inference"
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
                    conditioning=reference,
                    class_ids=class_ids,
                )
                reconstruction = self.diffusers_adapter.reconstruct_clean(noisy_image, predicted_noise, timesteps)
            else:
                noisy_image, target_noise, timesteps = self._sample_noisy_inputs(
                    clean_image,
                    inference=True,
                    target_noise=target_noise,
                )
                conditional_inputs = torch.cat([noisy_image, reference], dim=1)
                predicted_noise = self.conditional_denoiser(conditional_inputs, timesteps, class_ids=class_ids)
                noise_scale = self._noise_scale_from_timesteps(timesteps).view(-1, 1, 1, 1)
                reconstruction = legacy_reconstruct_clean(noisy_image, predicted_noise, noise_scale.view(-1))
            reference_projection = reconstruction
        conditional_alignment = torch.abs(reference_projection - reference).mean(dim=1).squeeze(0)

        return build_conditional_reconstruction_artifacts(
            sample_id=sample.sample_id,
            category=sample.category,
            representation=self.serialize_representation(representation),
            reconstruction=reconstruction.squeeze(0),
            reference_projection=reference_projection.squeeze(0),
            predicted_noise=predicted_noise.squeeze(0),
            target_noise=target_noise.squeeze(0),
            conditional_alignment=conditional_alignment,
            diagnostics={
                "fit_loss": self.last_fit_loss,
                "noise_level": self.noise_level,
                "time_embed_dim": self.time_embed_dim,
                "num_train_timesteps": self.num_train_timesteps,
                "inference_timestep": int(timesteps[0].item()),
            },
        )

    def _sample_noisy_inputs(
        self,
        clean_batch: torch.Tensor,
        *,
        inference: bool = False,
        target_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample Gaussian noise and produce the corresponding noisy inputs."""

        noisy_batch, target_noise, timesteps, _noise_scales = sample_legacy_noisy_inputs(
            clean_batch,
            num_train_timesteps=self.num_train_timesteps,
            noise_level=self.noise_level,
            inference=inference,
            target_noise=target_noise,
        )
        return noisy_batch, target_noise, timesteps

    def _noise_scale_from_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Map discrete timesteps onto bounded per-sample noise scales."""

        timestep_fraction = (timesteps.float() + 1.0) / float(self.num_train_timesteps)
        return self.noise_level * torch.sqrt(timestep_fraction)

    def _ensure_diffusers_backend(self, sample_size: int) -> None:
        """Instantiate the reference-conditioned diffusers backend lazily."""

        if self.backend != "diffusers" or self.diffusers_adapter is not None:
            return
        self.diffusers_adapter = DiffusersNoisePredictorAdapter(
            input_channels=self.input_channels * 2,
            output_channels=self.input_channels,
            hidden_channels=self.base_channels,
            learning_rate=self.learning_rate,
            noise_level=self.noise_level,
            sample_size=sample_size,
            num_train_timesteps=self.num_train_timesteps,
            num_classes=self.num_classes,
        )
        self.diffusers_adapter.to(self.runtime.device)
        install_normality_runtime_state(self.diffusers_adapter, self.runtime)

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
            needs_unit_rescale = not torch.is_floating_point(reference)
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
            if needs_unit_rescale:
                reference_tensor = reference_tensor / 255.0
                reference_tensor = self._apply_reference_normalization(sample, reference_tensor)
            return reference_tensor.to(dtype=image.dtype, device=image.device)

        if isinstance(reference, Image.Image):
            resized = tv_functional.resize(
                reference,
                target_size,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            reference_tensor = tv_functional.to_tensor(resized)
            reference_tensor = self._apply_reference_normalization(sample, reference_tensor)
            return reference_tensor.to(dtype=image.dtype, device=image.device)

        raise TypeError(
            "ReferenceDiffusionBasicNormality expects sample.reference to be a PIL image or torch.Tensor."
        )

    @staticmethod
    def _apply_reference_normalization(sample: Sample, reference_tensor: torch.Tensor) -> torch.Tensor:
        """Apply the sample-level normalization policy to a raw reference tensor when available."""

        if not sample.metadata.get("_transform_normalize"):
            return reference_tensor
        mean = sample.metadata.get("_transform_mean")
        std = sample.metadata.get("_transform_std")
        if not isinstance(mean, list) or not isinstance(std, list):
            return reference_tensor
        return tv_functional.normalize(reference_tensor, tuple(float(value) for value in mean), tuple(float(value) for value in std))
