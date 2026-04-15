"""Minimal reference-conditioned normality model."""

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
from adrf.normality.diffusion_basic import _ResidualConvBlock, _normalize_channel_mults
from adrf.normality.base import BaseNormalityModel
from adrf.normality.state import make_default_normality_runtime_state
from adrf.representation.contracts import RepresentationOutput


class _ConditionalProjector(nn.Module):
    """Configurable conditional projector with explicit fusion width."""

    def __init__(
        self,
        input_channels: int,
        base_channels: int,
        condition_channels: int,
        channel_mults: Sequence[int],
        num_res_blocks: int,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.image_projection = nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1)
        self.condition_projection = nn.Conv2d(input_channels, condition_channels, kernel_size=3, padding=1)
        layers: list[nn.Module] = []
        current_channels = base_channels + condition_channels
        for multiplier in channel_mults:
            stage_channels = base_channels * int(multiplier)
            layers.append(_ResidualConvBlock(current_channels, stage_channels))
            current_channels = stage_channels
            for _ in range(num_res_blocks - 1):
                layers.append(_ResidualConvBlock(current_channels, current_channels))
        self.backbone = nn.Sequential(*layers)
        self.output_projection = nn.Conv2d(current_channels, input_channels, kernel_size=3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project one concatenated image/reference tensor back into image space."""

        image, reference = torch.split(inputs, [self.input_channels, inputs.shape[1] - self.input_channels], dim=1)
        fused = torch.cat(
            [
                self.image_projection(image),
                self.condition_projection(reference),
            ],
            dim=1,
        )
        return self.output_projection(self.backbone(fused))


class ReferenceBasicNormality(nn.Module, BaseNormalityModel):
    """Learn a minimal reference-conditioned projection in pixel space."""

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
        learning_rate: float = 1e-3,
        epochs: int = 1,
        batch_size: int = 8,
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
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.base_channels = resolved_base_channels
        self.hidden_channels = resolved_base_channels
        self.condition_channels = resolved_condition_channels
        self.channel_mults = _normalize_channel_mults(channel_mults)
        self.num_res_blocks = int(num_res_blocks)
        self.conditional_model = _ConditionalProjector(
            input_channels=input_channels,
            base_channels=self.base_channels,
            condition_channels=self.condition_channels,
            channel_mults=self.channel_mults,
            num_res_blocks=self.num_res_blocks,
        )
        self.last_fit_loss: float | None = None
        self.runtime = make_default_normality_runtime_state()
        self.runtime_device = self.runtime.device
        self.amp_enabled = self.runtime.amp_enabled
        self.grad_scaler = self.runtime.grad_scaler
        self.distributed_context = self.runtime.distributed_context
        self.distributed_training_enabled = self.runtime.distributed_training_enabled
        self.eval()

    def fit(
        self,
        representations: Iterable[RepresentationOutput],
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

    def infer(
        self,
        sample: Sample,
        representation: RepresentationOutput,
    ) -> NormalityArtifacts:
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
            representation=self.serialize_representation(representation),
            primary={"reference_projection": reference_projection},
            auxiliary={"conditional_alignment": conditional_alignment},
            diagnostics={"fit_loss": self.last_fit_loss},
            capabilities={"reference_projection", "conditional_alignment"},
        )

    def _prepare_reference_tensor(
        self,
        sample: Sample,
        representation: RepresentationOutput,
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
