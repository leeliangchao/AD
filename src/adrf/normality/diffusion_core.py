"""Shared helpers for diffusion-style normality models."""

from __future__ import annotations

import hashlib
import math
from typing import Sequence

import torch


def normalize_channel_mults(
    channel_mults: Sequence[int] | None,
    *,
    default: tuple[int, ...] = (1,),
) -> tuple[int, ...]:
    """Normalize a channel multiplier sequence into a validated tuple."""

    if channel_mults is None:
        return default
    normalized = tuple(int(multiplier) for multiplier in channel_mults)
    if not normalized or any(multiplier < 1 for multiplier in normalized):
        raise ValueError("channel_mults must be a non-empty sequence of positive integers.")
    return normalized


def validate_diffusion_backend(
    backend: str,
    *,
    supported_backends: Sequence[str],
    model_name: str,
) -> str:
    """Validate one diffusion backend string against the supported set."""

    normalized = str(backend).strip().lower()
    supported = tuple(str(value).strip().lower() for value in supported_backends)
    if normalized not in supported:
        supported_text = ", ".join(supported)
        raise ValueError(f"Unsupported diffusion backend `{backend}` for {model_name}. Supported backends: {supported_text}.")
    return normalized


def deterministic_noise_like(tensor: torch.Tensor, *identity_parts: object) -> torch.Tensor:
    """Sample deterministic Gaussian noise from a stable identity tuple."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_identity_seed(*identity_parts))
    noise = torch.randn(tensor.shape, generator=generator, dtype=torch.float32, device="cpu")
    return noise.to(device=tensor.device, dtype=tensor.dtype)


def stable_identity_seed(*parts: object) -> int:
    """Map arbitrary identity parts onto one stable integer seed."""

    payload = "||".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**31)


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    """Create sinusoidal timestep embeddings for diffusion conditioning."""

    if embedding_dim < 1:
        raise ValueError("embedding_dim must be at least 1.")

    half_dim = embedding_dim // 2
    if half_dim == 0:
        return timesteps.float().unsqueeze(1)

    exponent = -math.log(max_period) * torch.arange(
        half_dim,
        dtype=torch.float32,
        device=timesteps.device,
    ) / max(half_dim - 1, 1)
    frequencies = torch.exp(exponent)
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if embedding_dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
    return embedding


def legacy_noise_scale_from_timesteps(
    timesteps: torch.Tensor,
    *,
    num_train_timesteps: int,
    noise_level: float,
) -> torch.Tensor:
    """Map discrete timesteps onto bounded per-sample noise scales."""

    timestep_fraction = (timesteps.float() + 1.0) / float(num_train_timesteps)
    return noise_level * torch.sqrt(timestep_fraction)


def sample_legacy_timesteps(
    *,
    batch_size: int,
    device: torch.device,
    num_train_timesteps: int,
    inference: bool = False,
) -> torch.Tensor:
    """Sample training timesteps or choose a deterministic inference timestep."""

    if inference:
        return torch.full(
            (batch_size,),
            fill_value=num_train_timesteps - 1,
            dtype=torch.long,
            device=device,
        )
    return torch.randint(
        low=0,
        high=num_train_timesteps,
        size=(batch_size,),
        dtype=torch.long,
        device=device,
    )


def sample_legacy_noisy_inputs(
    clean_batch: torch.Tensor,
    *,
    num_train_timesteps: int,
    noise_level: float,
    inference: bool = False,
    target_noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample Gaussian noise and produce the corresponding noisy inputs."""

    timesteps = sample_legacy_timesteps(
        batch_size=clean_batch.shape[0],
        device=clean_batch.device,
        num_train_timesteps=num_train_timesteps,
        inference=inference,
    )
    noise_scales = legacy_noise_scale_from_timesteps(
        timesteps,
        num_train_timesteps=num_train_timesteps,
        noise_level=noise_level,
    )
    if target_noise is None:
        target_noise = torch.randn_like(clean_batch)
    noisy_batch = clean_batch + noise_scales.view(-1, 1, 1, 1) * target_noise
    return noisy_batch, target_noise, timesteps, noise_scales


def legacy_reconstruct_clean(
    noisy_batch: torch.Tensor,
    predicted_noise: torch.Tensor,
    noise_scales: torch.Tensor,
) -> torch.Tensor:
    """Estimate x0 from a legacy noisy input and unscaled predicted noise."""

    return noisy_batch - noise_scales.view(-1, 1, 1, 1) * predicted_noise


def run_legacy_reverse_rollout(
    initial_state: torch.Tensor,
    rollout_timesteps: torch.Tensor,
    rollout_scales: torch.Tensor,
    *,
    predict_noise_fn,
    step_size: float,
    rollout_gain: float,
    denoised_blend: float,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Run the shared legacy-style reverse rollout and return canonical process artifacts."""

    current_state = initial_state
    trajectory: list[torch.Tensor] = []
    step_updates: list[torch.Tensor] = []
    step_costs: list[torch.Tensor] = []

    for step_index, timestep in enumerate(rollout_timesteps):
        timestep_batch = timestep.expand(current_state.shape[0])
        current_noise_scale = rollout_scales[step_index].expand(current_state.shape[0])
        predicted_noise = predict_noise_fn(current_state, timestep_batch, current_noise_scale)
        trajectory.append(current_state.squeeze(0).detach().clone())
        if step_index < len(rollout_timesteps) - 1:
            rollout_scale = rollout_scales[step_index].view(-1, 1, 1, 1)
            next_scale = rollout_scales[step_index + 1].view(-1, 1, 1, 1)
            predicted_clean = legacy_reconstruct_clean(current_state, predicted_noise, rollout_scale.view(-1))
            scale_delta = (rollout_scale - next_scale).clamp_min(0.0)
            next_state = current_state - step_size * rollout_gain * scale_delta * predicted_noise
            if denoised_blend > 0:
                next_state = (1.0 - denoised_blend) * next_state + denoised_blend * predicted_clean
            step_update = (next_state - current_state).squeeze(0).detach().clone()
            step_updates.append(step_update)
            step_costs.append(step_update.abs().mean(dim=0))
            current_state = next_state
        else:
            zero_update = torch.zeros_like(current_state.squeeze(0))
            step_updates.append(zero_update)
            step_costs.append(torch.zeros_like(current_state.mean(dim=1).squeeze(0)))

    return current_state, trajectory, step_updates, step_costs
