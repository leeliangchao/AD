"""Minimal convolutional autoencoder normality model."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Sequence
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample
from adrf.normality.diffusion_basic import _normalize_channel_mults
from adrf.normality.base import BaseNormalityModel


class _ConvStack(nn.Module):
    """A configurable conv stack used for stronger encoder/decoder stages."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        ]
        for _ in range(num_blocks - 1):
            layers.extend(
                [
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.SiLU(inplace=True),
                ]
            )
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the stage conv stack."""

        return self.network(inputs)


class AutoEncoderNormality(nn.Module, BaseNormalityModel):
    """Train a small convolutional autoencoder on normal pixel representations."""

    def __init__(
        self,
        input_channels: int = 3,
        hidden_channels: int = 16,
        base_channels: int | None = None,
        channel_mults: Sequence[int] | None = None,
        latent_channels: int = 32,
        num_blocks_per_stage: int = 1,
        learning_rate: float = 1e-3,
        epochs: int = 1,
        batch_size: int = 8,
    ) -> None:
        super().__init__()
        resolved_base_channels = int(base_channels if base_channels is not None else hidden_channels)
        if resolved_base_channels < 1:
            raise ValueError("base_channels must be at least 1.")
        if latent_channels < 1:
            raise ValueError("latent_channels must be at least 1.")
        if num_blocks_per_stage < 1:
            raise ValueError("num_blocks_per_stage must be at least 1.")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.base_channels = resolved_base_channels
        self.hidden_channels = resolved_base_channels
        self.channel_mults = _normalize_channel_mults(channel_mults)
        self.latent_channels = int(latent_channels)
        self.num_blocks_per_stage = int(num_blocks_per_stage)

        encoder_channels = [self.base_channels * multiplier for multiplier in self.channel_mults]
        encoder_layers: list[nn.Module] = []
        current_channels = input_channels
        for stage_channels in encoder_channels:
            encoder_layers.append(_ConvStack(current_channels, stage_channels, self.num_blocks_per_stage))
            encoder_layers.append(nn.Conv2d(stage_channels, stage_channels, kernel_size=4, stride=2, padding=1))
            encoder_layers.append(nn.SiLU(inplace=True))
            current_channels = stage_channels
        encoder_layers.append(nn.Conv2d(current_channels, self.latent_channels, kernel_size=3, padding=1))
        encoder_layers.append(nn.SiLU(inplace=True))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: list[nn.Module] = []
        current_channels = self.latent_channels
        for stage_channels in reversed(encoder_channels):
            decoder_layers.append(nn.ConvTranspose2d(current_channels, stage_channels, kernel_size=4, stride=2, padding=1))
            decoder_layers.append(nn.SiLU(inplace=True))
            decoder_layers.append(_ConvStack(stage_channels, stage_channels, self.num_blocks_per_stage))
            current_channels = stage_channels
        decoder_layers.append(nn.Conv2d(current_channels, input_channels, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*decoder_layers)
        self.last_fit_loss: float | None = None
        self.eval()

    def fit(
        self,
        representations: Iterable[Mapping[str, Any]],
        samples: Iterable[Sample] | None = None,
    ) -> None:
        """Train the autoencoder on pixel-space representations."""

        del samples
        tensors = [self.require_representation_tensor(representation).float() for representation in representations]
        if not tensors:
            raise ValueError("AutoEncoderNormality.fit requires at least one representation.")

        train_batch = torch.stack(tensors, dim=0)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.train()
        for _ in range(self.epochs):
            permutation = torch.randperm(train_batch.shape[0])
            shuffled = train_batch[permutation]
            for start in range(0, shuffled.shape[0], self.batch_size):
                batch = shuffled[start : start + self.batch_size]
                projection, reconstruction = self._forward_impl(batch)
                del projection
                loss = functional.mse_loss(reconstruction, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.last_fit_loss = float(loss.detach().cpu().item())
        self.eval()

    def infer(self, sample: Sample, representation: Mapping[str, Any]) -> NormalityArtifacts:
        """Run the trained autoencoder and emit projection/reconstruction artifacts."""

        image = self.require_representation_tensor(representation).float().unsqueeze(0)
        with torch.no_grad():
            projection, reconstruction = self._forward_impl(image)
        return NormalityArtifacts(
            context={"sample_id": sample.sample_id, "category": sample.category},
            representation=dict(representation),
            primary={
                "projection": projection.squeeze(0),
                "reconstruction": reconstruction.squeeze(0),
            },
            diagnostics={"fit_loss": self.last_fit_loss},
            capabilities={"projection", "reconstruction"},
        )

    def _forward_impl(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode and decode a batch of image tensors."""

        projection = self.encoder(inputs)
        reconstruction = self.decoder(projection)
        return projection, reconstruction
