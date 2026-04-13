"""Minimal reference-conditioned diffusion normality model."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.nn.functional as functional
from PIL import Image
from torch import nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tv_functional

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample
from adrf.normality.base import BaseNormalityModel


class ReferenceDiffusionBasicNormality(nn.Module, BaseNormalityModel):
    """Predict diffusion noise under a fixed per-category reference condition."""

    def __init__(
        self,
        input_channels: int = 3,
        hidden_channels: int = 16,
        learning_rate: float = 1e-3,
        epochs: int = 1,
        batch_size: int = 8,
        noise_level: float = 0.2,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.noise_level = noise_level
        self.conditional_denoiser = nn.Sequential(
            nn.Conv2d(input_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, input_channels, kernel_size=3, padding=1),
        )
        self.last_fit_loss: float | None = None
        self.eval()

    def fit(
        self,
        representations: Iterable[Mapping[str, Any]],
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
                noisy_batch, target_noise = self._sample_noisy_inputs(clean_batch)
                conditional_inputs = torch.cat([noisy_batch, reference_slice], dim=1)
                predicted_noise = self.conditional_denoiser(conditional_inputs)
                loss = functional.mse_loss(predicted_noise, target_noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.last_fit_loss = float(loss.detach().cpu().item())
        self.eval()

    def infer(self, sample: Sample, representation: Mapping[str, Any]) -> NormalityArtifacts:
        """Infer conditional diffusion artifacts for one sample."""

        clean_image = self.require_representation_tensor(representation).float().unsqueeze(0)
        reference = self._prepare_reference_tensor(sample, representation).float().unsqueeze(0)
        with torch.no_grad():
            noisy_image, target_noise = self._sample_noisy_inputs(clean_image)
            conditional_inputs = torch.cat([noisy_image, reference], dim=1)
            predicted_noise = self.conditional_denoiser(conditional_inputs)
            reference_projection = noisy_image - self.noise_level * predicted_noise
        conditional_alignment = torch.abs(reference_projection - reference).mean(dim=1).squeeze(0)

        return NormalityArtifacts(
            context={
                "sample_id": sample.sample_id,
                "category": sample.category,
                "has_reference": sample.has_reference(),
                "mode": "inference",
            },
            representation={
                "space_type": representation.get("space_type"),
                "spatial_shape": representation.get("spatial_shape"),
            },
            primary={"reference_projection": reference_projection.squeeze(0)},
            auxiliary={
                "predicted_noise": predicted_noise.squeeze(0),
                "target_noise": target_noise.squeeze(0),
                "conditional_alignment": conditional_alignment,
            },
            diagnostics={
                "fit_loss": self.last_fit_loss,
                "noise_level": self.noise_level,
            },
            capabilities={
                "predicted_noise",
                "target_noise",
                "reference_projection",
                "conditional_alignment",
            },
        )

    def _sample_noisy_inputs(self, clean_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample Gaussian noise and produce the corresponding noisy inputs."""

        target_noise = torch.randn_like(clean_batch)
        noisy_batch = clean_batch + self.noise_level * target_noise
        return noisy_batch, target_noise

    def _prepare_reference_tensor(
        self,
        sample: Sample,
        representation: Mapping[str, Any],
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
