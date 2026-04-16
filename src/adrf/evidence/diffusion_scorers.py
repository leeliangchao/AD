"""Shared scoring helpers for diffusion-family artifacts."""

from __future__ import annotations

import torch


def score_reconstruction_residual(image: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    if image.shape != reconstruction.shape:
        raise ValueError("image and reconstruction must have the same shape.")
    return torch.abs(image.float() - reconstruction.float()).mean(dim=0)


def score_noise_residual(predicted_noise: torch.Tensor, target_noise: torch.Tensor) -> torch.Tensor:
    if predicted_noise.shape != target_noise.shape:
        raise ValueError("predicted_noise and target_noise must have the same shape.")
    return torch.abs(predicted_noise.float() - target_noise.float()).mean(dim=0)


def score_conditional_residual(image: torch.Tensor, reference_projection: torch.Tensor) -> torch.Tensor:
    if image.shape != reference_projection.shape:
        raise ValueError("image and reference_projection must have the same shape.")
    return torch.abs(image.float() - reference_projection.float()).mean(dim=0)


def score_path_cost_from_step_costs(step_costs: list[torch.Tensor]) -> torch.Tensor:
    normalized = [_normalize_step_cost(cost_map) for cost_map in step_costs]
    return torch.stack(normalized, dim=0).sum(dim=0)


def score_path_cost_from_step_updates(step_updates: list[torch.Tensor]) -> torch.Tensor:
    normalized = [_normalize_step_update(update) for update in step_updates]
    return torch.stack(normalized, dim=0).sum(dim=0)


def score_direction_mismatch_from_step_updates(
    step_updates: list[torch.Tensor],
    *,
    direction_reduce: str,
) -> torch.Tensor:
    if len(step_updates) < 2:
        raise ValueError("Direction mismatch requires at least two step updates.")
    mismatch_maps = []
    transitions = [update.float() for update in step_updates]
    for previous_delta, current_delta in zip(transitions[:-1], transitions[1:], strict=True):
        reversal = torch.relu(-(previous_delta * current_delta))
        mismatch_maps.append(_reduce_direction_channels(reversal))
    stacked = torch.stack(mismatch_maps, dim=0)
    if direction_reduce == "sum":
        return stacked.sum(dim=0)
    if direction_reduce == "mean":
        return stacked.mean(dim=0)
    raise ValueError("direction_reduce must be either 'sum' or 'mean'.")


def score_latent_reconstruction_residual(
    latent_reconstruction: torch.Tensor,
    latent_target: torch.Tensor,
) -> torch.Tensor:
    if latent_reconstruction.shape != latent_target.shape:
        raise ValueError("latent_reconstruction and latent_target must have the same shape.")
    return torch.abs(latent_reconstruction.float() - latent_target.float()).mean(dim=0)


def score_passthrough_score_map(score_map: torch.Tensor) -> torch.Tensor:
    if score_map.ndim != 2:
        raise ValueError("score_map must already be a 2D anomaly map.")
    return score_map.float()


def _normalize_step_cost(step_cost: torch.Tensor) -> torch.Tensor:
    if not isinstance(step_cost, torch.Tensor):
        raise TypeError("Each step cost must be a torch.Tensor.")
    if step_cost.ndim == 2:
        return step_cost.float()
    if step_cost.ndim == 3 and step_cost.shape[0] == 1:
        return step_cost.squeeze(0).float()
    if step_cost.ndim == 3:
        return step_cost.float().mean(dim=0)
    raise ValueError("Step cost tensors must be 2D or 3D.")


def _normalize_step_update(step_update: torch.Tensor) -> torch.Tensor:
    if not isinstance(step_update, torch.Tensor):
        raise TypeError("Each step update must be a torch.Tensor.")
    if step_update.ndim == 2:
        return step_update.abs().float()
    if step_update.ndim == 3 and step_update.shape[0] == 1:
        return step_update.squeeze(0).abs().float()
    if step_update.ndim == 3:
        return step_update.abs().float().mean(dim=0)
    raise ValueError("Step update tensors must be 2D or 3D.")


def _reduce_direction_channels(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor.mean(dim=0)
    if tensor.ndim == 2:
        return tensor
    raise ValueError("Trajectory states must be 2D or 3D tensors.")
