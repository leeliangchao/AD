"""Minimal process-style diffusion normality model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Sequence
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
        base_channels: int | None = None,
        channel_mults: Sequence[int] | None = None,
        num_res_blocks: int = 2,
        time_embed_dim: int = 64,
        num_train_timesteps: int = 100,
        learning_rate: float = 1e-3,
        epochs: int = 1,
        batch_size: int = 8,
        noise_level: float = 0.2,
        num_steps: int = 4,
        step_size: float = 0.1,
        initial_noise_scale: float | None = None,
        backend: str = "legacy",
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            time_embed_dim=time_embed_dim,
            num_train_timesteps=num_train_timesteps,
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
        if initial_noise_scale is not None and initial_noise_scale <= 0:
            raise ValueError("initial_noise_scale must be positive when provided.")
        self.num_steps = num_steps
        self.step_size = step_size
        self.initial_noise_scale = float(initial_noise_scale) if initial_noise_scale is not None else noise_level

    def infer(self, sample: Sample, representation: Mapping[str, Any]) -> NormalityArtifacts:
        """Run a fixed denoising trajectory and expose step-aligned process artifacts."""

        current_state = self._initial_state(representation)
        trajectory: list[torch.Tensor] = []
        step_costs: list[torch.Tensor] = []

        with torch.no_grad():
            rollout_timesteps = self._rollout_timesteps(current_state.device)
            for step_index, timestep in enumerate(rollout_timesteps):
                timestep_batch = timestep.expand(current_state.shape[0])
                if self.backend == "diffusers":
                    self._ensure_diffusers_backend(sample_size=int(current_state.shape[-1]))
                    predicted_noise = self.diffusers_adapter.model(current_state, timestep_batch)
                else:
                    predicted_noise = self.denoiser(current_state, timestep_batch)
                trajectory.append(current_state.squeeze(0).detach().clone())
                step_costs.append(predicted_noise.abs().mean(dim=1).squeeze(0).detach().clone())
                if step_index < self.num_steps - 1:
                    rollout_scale = self._noise_scale_from_timesteps(timestep_batch).view(-1, 1, 1, 1)
                    current_state = current_state - self.step_size * rollout_scale * predicted_noise

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
                "initial_noise_scale": self.initial_noise_scale,
                "time_embed_dim": self.time_embed_dim,
                "num_train_timesteps": self.num_train_timesteps,
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
        perturbation = self.initial_noise_scale * torch.randn_like(clean_image)
        return clean_image + perturbation

    def _rollout_timesteps(self, device: torch.device) -> torch.Tensor:
        """Create a descending timestep schedule for the finite denoising process."""

        if self.num_steps == 1:
            return torch.tensor([self.num_train_timesteps - 1], dtype=torch.long, device=device)
        return torch.linspace(
            self.num_train_timesteps - 1,
            0,
            steps=self.num_steps,
            device=device,
        ).round().long()
