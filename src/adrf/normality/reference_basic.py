"""Minimal reference-conditioned normality model."""

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


class ReferenceBasicNormality(nn.Module, BaseNormalityModel):
    """Learn a minimal reference-conditioned projection in pixel space."""

    def __init__(
        self,
        input_channels: int = 3,
        hidden_channels: int = 16,
        learning_rate: float = 1e-3,
        epochs: int = 1,
        batch_size: int = 8,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.conditional_model = nn.Sequential(
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
        """Train the conditional model to reconstruct normal images given a reference."""

        if samples is None:
            raise ValueError("ReferenceBasicNormality.fit requires samples with reference inputs.")

        paired_tensors = [
            (
                self.require_representation_tensor(representation).float(),
                self._prepare_reference_tensor(sample, representation).float(),
            )
            for representation, sample in zip(representations, samples, strict=True)
        ]
        if not paired_tensors:
            raise ValueError("ReferenceBasicNormality.fit requires at least one representation.")

        image_batch = torch.stack([image for image, _ in paired_tensors], dim=0)
        reference_batch = torch.stack([reference for _, reference in paired_tensors], dim=0)
        conditional_batch = torch.cat([image_batch, reference_batch], dim=1)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.train()
        for _ in range(self.epochs):
            permutation = torch.randperm(conditional_batch.shape[0])
            shuffled_inputs = conditional_batch[permutation]
            shuffled_targets = image_batch[permutation]
            for start in range(0, shuffled_inputs.shape[0], self.batch_size):
                batch_inputs = shuffled_inputs[start : start + self.batch_size]
                batch_targets = shuffled_targets[start : start + self.batch_size]
                projection = self.conditional_model(batch_inputs)
                loss = functional.mse_loss(projection, batch_targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.last_fit_loss = float(loss.detach().cpu().item())
        self.eval()

    def infer(self, sample: Sample, representation: Mapping[str, Any]) -> NormalityArtifacts:
        """Infer a minimal conditional projection and alignment response."""

        image = self.require_representation_tensor(representation).float()
        reference = self._prepare_reference_tensor(sample, representation).float()
        conditional_input = torch.cat([image, reference], dim=0).unsqueeze(0)
        with torch.no_grad():
            reference_projection = self.conditional_model(conditional_input).squeeze(0)
        conditional_alignment = torch.abs(reference_projection - reference).mean(dim=0)

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
            primary={"reference_projection": reference_projection},
            auxiliary={"conditional_alignment": conditional_alignment},
            diagnostics={"fit_loss": self.last_fit_loss},
            capabilities={"reference_projection", "conditional_alignment"},
        )

    def _prepare_reference_tensor(
        self,
        sample: Sample,
        representation: Mapping[str, Any],
    ) -> torch.Tensor:
        """Convert the sample reference into a tensor aligned with the image representation."""

        if sample.reference is None:
            raise ValueError("ReferenceBasicNormality requires sample.reference to be present.")

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

        raise TypeError("ReferenceBasicNormality expects sample.reference to be a PIL image or torch.Tensor.")
