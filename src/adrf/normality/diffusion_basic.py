"""Minimal diffusion-style normality model based on single-step noise prediction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample
from adrf.diffusion.adapters import DiffusersNoisePredictorAdapter
from adrf.normality.base import BaseNormalityModel


class DiffusionBasicNormality(nn.Module, BaseNormalityModel):
    """Train a small denoiser on normal pixel samples and expose noise artifacts."""

    def __init__(
        self,
        input_channels: int = 3,
        hidden_channels: int = 16,
        learning_rate: float = 1e-3,
        epochs: int = 1,
        batch_size: int = 8,
        noise_level: float = 0.2,
        backend: str = "legacy",
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.noise_level = noise_level
        self.backend = backend
        self.denoiser = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, input_channels, kernel_size=3, padding=1),
        )
        self.diffusers_adapter: DiffusersNoisePredictorAdapter | None = None
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.last_fit_loss: float | None = None
        self.runtime_device = torch.device("cpu")
        self.amp_enabled = False
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=False)
        self._adrf_runtime_wrapped = True
        self.eval()

    def fit(
        self,
        representations: Iterable[Mapping[str, Any]],
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
                        noisy_batch, target_noise = self._sample_noisy_inputs(clean_batch)
                        predicted_noise = self.denoiser(noisy_batch)
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

    def infer(self, sample: Sample, representation: Mapping[str, Any]) -> NormalityArtifacts:
        """Run one single-step noise-prediction pass and emit diffusion artifacts."""

        clean_image = self.require_representation_tensor(representation).float().unsqueeze(0)
        with torch.no_grad():
            if self.backend == "diffusers":
                self._ensure_diffusers_backend(sample_size=int(clean_image.shape[-1]))
                predicted_noise, target_noise = self.diffusers_adapter.forward_infer_step(clean_image)
            else:
                noisy_image, target_noise = self._sample_noisy_inputs(clean_image)
                predicted_noise = self.denoiser(noisy_image)
        return NormalityArtifacts(
            context={
                "sample_id": sample.sample_id,
                "category": sample.category,
                "mode": "inference",
            },
            representation={
                "space_type": representation.get("space_type"),
                "spatial_shape": representation.get("spatial_shape"),
            },
            primary={},
            auxiliary={
                "predicted_noise": predicted_noise.squeeze(0),
                "target_noise": target_noise.squeeze(0),
            },
            diagnostics={
                "fit_loss": self.last_fit_loss,
                "noise_level": self.noise_level,
            },
            capabilities={"predicted_noise", "target_noise"},
        )

    def _sample_noisy_inputs(self, clean_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample Gaussian noise and produce the corresponding noisy inputs."""

        target_noise = torch.randn_like(clean_batch)
        noisy_batch = clean_batch + self.noise_level * target_noise
        return noisy_batch, target_noise

    def _ensure_diffusers_backend(self, sample_size: int) -> None:
        """Instantiate the diffusers backend lazily when first needed."""

        if self.backend != "diffusers" or self.diffusers_adapter is not None:
            return
        self.diffusers_adapter = DiffusersNoisePredictorAdapter(
            input_channels=self.input_channels,
            hidden_channels=self.hidden_channels,
            learning_rate=self.learning_rate,
            noise_level=self.noise_level,
            sample_size=sample_size,
        )
        self.diffusers_adapter.to(self.runtime_device)
        self.diffusers_adapter.amp_enabled = self.amp_enabled
        self.diffusers_adapter.grad_scaler = self.grad_scaler
