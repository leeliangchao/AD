"""Minimal process-style diffusion normality model."""

from __future__ import annotations

from typing import Sequence

import torch

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample
from adrf.normality.diffusion_basic import DiffusionBasicNormality
from adrf.normality.diffusion_conditioning import resolve_optional_class_ids
from adrf.normality.diffusion_core import deterministic_noise_like, run_legacy_reverse_rollout
from adrf.normality.diffusion_tasks import build_trajectory_artifacts
from adrf.representation.contracts import RepresentationOutput


class DiffusionInversionBasicNormality(DiffusionBasicNormality):
    """Run a finite denoising process and expose the full state trajectory."""

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
        num_steps: int = 4,
        step_size: float = 0.1,
        initial_noise_scale: float | None = None,
        rollout_gain: float = 1.0,
        denoised_blend: float = 0.0,
        backend: str = "legacy",
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            time_embed_dim=time_embed_dim,
            conditioning_hidden_dim=conditioning_hidden_dim,
            num_train_timesteps=num_train_timesteps,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            noise_level=noise_level,
            num_classes=num_classes,
            class_embed_dim=class_embed_dim,
            backend=backend,
        )
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1.")
        if step_size <= 0:
            raise ValueError("step_size must be positive.")
        if initial_noise_scale is not None and initial_noise_scale <= 0:
            raise ValueError("initial_noise_scale must be positive when provided.")
        if rollout_gain <= 0:
            raise ValueError("rollout_gain must be positive.")
        if not 0.0 <= denoised_blend <= 1.0:
            raise ValueError("denoised_blend must be between 0 and 1.")
        self.num_steps = num_steps
        self.step_size = step_size
        self.initial_noise_scale = float(initial_noise_scale) if initial_noise_scale is not None else noise_level
        self.rollout_gain = float(rollout_gain)
        self.denoised_blend = float(denoised_blend)

    def infer(
        self,
        sample: Sample,
        representation: RepresentationOutput,
    ) -> NormalityArtifacts:
        """Run a fixed denoising trajectory and expose step-aligned process artifacts."""

        inference_identity = sample.sample_id or representation.sample_id or "inference"
        current_state = self._initial_state(representation, identity=inference_identity)
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
            class_ids = class_ids.to(device=current_state.device)

        with torch.no_grad():
            rollout_timesteps = self._rollout_timesteps(current_state.device)
            rollout_scales = self._noise_scale_from_timesteps(rollout_timesteps)
            if self.backend == "diffusers":
                self._ensure_diffusers_backend(sample_size=int(current_state.shape[-1]))

                def predict_noise_fn(state: torch.Tensor, timestep_batch: torch.Tensor, _noise_scale: torch.Tensor) -> torch.Tensor:
                    return self.diffusers_adapter.model(state, timestep_batch, class_ids=class_ids)
            else:

                def predict_noise_fn(state: torch.Tensor, timestep_batch: torch.Tensor, noise_scale: torch.Tensor) -> torch.Tensor:
                    if class_ids is None:
                        return self.denoiser(state, timestep_batch, noise_scale)
                    return self.denoiser(state, timestep_batch, noise_scale, class_ids=class_ids)

            reconstruction, trajectory, step_updates, step_costs = run_legacy_reverse_rollout(
                current_state,
                rollout_timesteps,
                rollout_scales,
                predict_noise_fn=predict_noise_fn,
                step_size=self.step_size,
                rollout_gain=self.rollout_gain,
                denoised_blend=self.denoised_blend,
            )

        final_state = reconstruction.squeeze(0)
        return build_trajectory_artifacts(
            sample_id=sample.sample_id,
            category=sample.category,
            representation=self.serialize_representation(representation),
            reconstruction=final_state,
            trajectory=trajectory,
            step_updates=step_updates,
            step_costs=step_costs,
            diagnostics={
                "fit_loss": self.last_fit_loss,
                "num_steps": self.num_steps,
                "initial_noise_scale": self.initial_noise_scale,
                "time_embed_dim": self.time_embed_dim,
                "conditioning_hidden_dim": self.conditioning_hidden_dim,
                "num_train_timesteps": self.num_train_timesteps,
                "rollout_gain": self.rollout_gain,
                "denoised_blend": self.denoised_blend,
                "final_state_norm": float(final_state.norm().item()),
                "step_summary": {
                    "first_step_cost_mean": float(step_costs[0].mean().item()),
                    "last_step_cost_mean": float(step_costs[-1].mean().item()),
                },
            },
        )

    def _initial_state(self, representation: RepresentationOutput, *, identity: object | None = None) -> torch.Tensor:
        """Build the initial perturbed state for inversion-style inference."""

        clean_image = self.require_representation_tensor(representation).float().unsqueeze(0)
        perturbation = self.initial_noise_scale * deterministic_noise_like(
            clean_image,
            type(self).__name__,
            identity,
            self.initial_noise_scale,
            self.num_steps,
        )
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
