"""Minimal convolutional autoencoder normality model."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample
from adrf.normality.base import BaseNormalityModel


class AutoEncoderNormality(nn.Module, BaseNormalityModel):
    """Train a small convolutional autoencoder on normal pixel representations."""

    def __init__(
        self,
        input_channels: int = 3,
        hidden_channels: int = 16,
        latent_channels: int = 32,
        learning_rate: float = 1e-3,
        epochs: int = 1,
        batch_size: int = 8,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                latent_channels,
                hidden_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                hidden_channels,
                input_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )
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
