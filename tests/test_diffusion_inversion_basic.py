"""Tests for the minimal process-style diffusion normality model."""

import torch

from adrf.core.sample import Sample
from adrf.normality.diffusion_inversion_basic import DiffusionInversionBasicNormality


def _pixel_representation(image: torch.Tensor) -> dict[str, object]:
    return {
        "representation": image,
        "space_type": "pixel",
        "spatial_shape": tuple(image.shape[-2:]),
    }


def test_diffusion_inversion_basic_emits_trajectory_and_step_costs() -> None:
    """DiffusionInversionBasicNormality should expose a process trace in artifacts."""

    torch.manual_seed(0)
    train_representations = [
        _pixel_representation(torch.rand(3, 16, 16)),
        _pixel_representation(torch.rand(3, 16, 16)),
    ]
    model = DiffusionInversionBasicNormality(
        input_channels=3,
        hidden_channels=8,
        learning_rate=1e-3,
        epochs=1,
        batch_size=2,
        noise_level=0.2,
        num_steps=4,
        step_size=0.1,
    )

    model.fit(train_representations)
    sample = Sample(image=train_representations[0]["representation"], sample_id="sample-001")
    artifacts = model.infer(sample, train_representations[0])

    assert artifacts.has("trajectory")
    assert artifacts.has("step_costs")
    assert "anomaly_map" not in artifacts.primary
    assert "image_score" not in artifacts.primary

    trajectory = artifacts.get_aux("trajectory")
    step_costs = artifacts.get_aux("step_costs")
    assert isinstance(trajectory, list)
    assert isinstance(step_costs, list)
    assert len(trajectory) == len(step_costs) == 4
    assert all(isinstance(state, torch.Tensor) for state in trajectory)
    assert all(isinstance(cost_map, torch.Tensor) for cost_map in step_costs)
    assert trajectory[0].shape == trajectory[-1].shape == (3, 16, 16)
    assert step_costs[0].shape == step_costs[-1].shape == (16, 16)

