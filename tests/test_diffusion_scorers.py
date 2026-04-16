"""Tests for internal diffusion scoring helpers."""

from __future__ import annotations

import torch

from adrf.evidence.diffusion_scorers import (
    score_conditional_residual,
    score_direction_mismatch_from_step_updates,
    score_noise_residual,
    score_path_cost_from_step_costs,
    score_path_cost_from_step_updates,
    score_reconstruction_residual,
)


def test_score_reconstruction_residual_returns_channel_reduced_l1_map() -> None:
    image = torch.tensor([[[0.0, 1.0], [0.5, 0.0]]], dtype=torch.float32)
    reconstruction = torch.tensor([[[0.0, 0.0], [0.25, 0.25]]], dtype=torch.float32)

    anomaly_map = score_reconstruction_residual(image, reconstruction)

    assert torch.allclose(anomaly_map, torch.tensor([[0.0, 1.0], [0.25, 0.25]], dtype=torch.float32))


def test_score_noise_residual_returns_channel_reduced_absolute_difference() -> None:
    predicted_noise = torch.tensor(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    target_noise = torch.zeros_like(predicted_noise)

    anomaly_map = score_noise_residual(predicted_noise, target_noise)

    assert torch.allclose(anomaly_map, torch.full((2, 2), 0.5, dtype=torch.float32))


def test_score_conditional_residual_matches_reference_projection_deviation() -> None:
    image = torch.tensor([[[0.0, 1.0], [0.5, 0.0]]], dtype=torch.float32)
    reference_projection = torch.tensor([[[0.0, 0.0], [0.25, 0.25]]], dtype=torch.float32)

    anomaly_map = score_conditional_residual(image, reference_projection)

    assert torch.allclose(anomaly_map, torch.tensor([[0.0, 1.0], [0.25, 0.25]], dtype=torch.float32))


def test_score_path_cost_from_step_costs_sums_cost_maps() -> None:
    step_costs = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32),
    ]

    anomaly_map = score_path_cost_from_step_costs(step_costs)

    assert torch.allclose(anomaly_map, torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float32))


def test_score_path_cost_from_step_updates_uses_absolute_update_magnitude() -> None:
    step_updates = [
        torch.tensor(
            [
                [[1.0, -2.0], [3.0, -4.0]],
                [[-1.0, 2.0], [-3.0, 4.0]],
            ],
            dtype=torch.float32,
        )
    ]

    anomaly_map = score_path_cost_from_step_updates(step_updates)

    assert torch.allclose(anomaly_map, torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32))


def test_score_direction_mismatch_from_step_updates_highlights_reversal() -> None:
    step_updates = [
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32),
        torch.tensor([[[-1.0, 0.0], [0.0, -1.0]]], dtype=torch.float32),
    ]

    anomaly_map = score_direction_mismatch_from_step_updates(step_updates, direction_reduce="sum")

    assert anomaly_map.shape == (2, 2)
    assert torch.all(anomaly_map >= 0)
