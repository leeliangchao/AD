"""Minimal process-style diffusion normality model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample
from adrf.normality.diffusion_basic import DiffusionBasicNormality


class DiffusionInversionBasicNormality(DiffusionBasicNormality):
    """Run a finite denoising process and expose the full state trajectory."""

    def __init__(
        self,
        input_channels: int = 3,
        hidden_channels: int = 16,
        learning_rate: float = 1e-3,
        epochs: int = 1,
        batch_size: int = 8,
        noise_level: float = 0.2,
        num_steps: int = 4,
        step_size: float = 0.1,
        backend: str = "legacy",
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            noise_level=noise_level,
            backend=backend,
        )
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1.")
        if step_size <= 0:
            raise ValueError("step_size must be positive.")
        self.num_steps = num_steps
        self.step_size = step_size

    def infer(self, sample: Sample, representation: Mapping[str, Any]) -> NormalityArtifacts:
        """Run a fixed denoising trajectory and expose step-aligned process artifacts."""

        current_state = self._initial_state(representation)
        trajectory: list[torch.Tensor] = []
        step_costs: list[torch.Tensor] = []

        with torch.no_grad():
            for step_index in range(self.num_steps):
                if self.backend == "diffusers":
                    self._ensure_diffusers_backend(sample_size=int(current_state.shape[-1]))
                    timesteps = torch.full(
                        (current_state.shape[0],),
                        fill_value=max(self.num_steps - step_index - 1, 0),
                        dtype=torch.long,
                        device=current_state.device,
                    )
                    predicted_noise = self.diffusers_adapter.model(current_state, timesteps)
                else:
                    predicted_noise = self.denoiser(current_state)
                trajectory.append(current_state.squeeze(0).detach().clone())
                step_costs.append(predicted_noise.abs().mean(dim=1).squeeze(0).detach().clone())
                if step_index < self.num_steps - 1:
                    current_state = current_state - self.step_size * predicted_noise

        final_state = trajectory[-1]
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
                "trajectory": trajectory,
                "step_costs": step_costs,
            },
            diagnostics={
                "fit_loss": self.last_fit_loss,
                "num_steps": self.num_steps,
                "final_state_norm": float(final_state.norm().item()),
                "step_summary": {
                    "first_step_cost_mean": float(step_costs[0].mean().item()),
                    "last_step_cost_mean": float(step_costs[-1].mean().item()),
                },
            },
            capabilities={"trajectory", "step_costs"},
        )

    def _initial_state(self, representation: Mapping[str, Any]) -> torch.Tensor:
        """Build the initial perturbed state for inversion-style inference."""

        clean_image = self.require_representation_tensor(representation).float().unsqueeze(0)
        perturbation = self.noise_level * torch.randn_like(clean_image)
        return clean_image + perturbation
