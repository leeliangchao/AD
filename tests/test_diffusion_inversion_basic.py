"""Tests for the minimal process-style diffusion normality model."""

from pathlib import Path
import sys

import torch

from adrf.core.sample import Sample
from adrf.normality.diffusion_inversion_basic import DiffusionInversionBasicNormality

sys.path.insert(0, str(Path(__file__).parent))

from support.representation_builders import make_pixel_output


def test_diffusion_inversion_fit_and_infer_accept_representation_output() -> None:
    """DiffusionInversionBasicNormality should expose process artifacts for typed pixel outputs."""

    generator = torch.Generator().manual_seed(0)
    representations = [
        make_pixel_output(torch.rand(3, 16, 16, generator=generator), sample_id=f"train-{index:03d}")
        for index in range(2)
    ]
    model = DiffusionInversionBasicNormality(
        input_channels=3,
        hidden_channels=8,
        time_embed_dim=32,
        num_train_timesteps=32,
        learning_rate=1e-3,
        epochs=1,
        batch_size=2,
        noise_level=0.2,
        num_steps=4,
        step_size=0.1,
    )

    model.fit(representations)
    sample = Sample(image=representations[0].tensor, sample_id="query")
    artifacts = model.infer(sample, representations[0])

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
    assert artifacts.get_diag("time_embed_dim") == 32
    assert artifacts.get_diag("num_train_timesteps") == 32
    assert artifacts.representation == representations[0].to_artifact_dict()
    assert artifacts.representation["space"] == "pixel"
    assert artifacts.representation["sample_id"] == "train-000"
    assert DiffusionInversionBasicNormality.accepted_spaces == frozenset({"pixel"})
    assert DiffusionInversionBasicNormality.accepted_tensor_ranks == frozenset({3})
