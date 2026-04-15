"""Tests for the process artifact contract of DiffusionInversionBasicNormality."""

from pathlib import Path
import sys

import torch
from torch import nn

from adrf.core.sample import Sample
from adrf.normality.diffusion_inversion_basic import DiffusionInversionBasicNormality

sys.path.insert(0, str(Path(__file__).parent))

from support.representation_builders import make_pixel_output


def test_diffusion_inversion_artifacts_preserve_step_aligned_contract() -> None:
    """Trajectory and step_costs should remain step-aligned under the diffusers backend."""

    generator = torch.Generator().manual_seed(0)
    train_representations = [
        make_pixel_output(torch.rand(3, 16, 16, generator=generator), sample_id=f"train-{index:03d}")
        for index in range(2)
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
        backend="diffusers",
    )
    model.fit(train_representations)
    sample = Sample(image=train_representations[0].tensor, sample_id="query")
    artifacts = model.infer(sample, train_representations[0])

    trajectory = artifacts.get_aux("trajectory")
    step_costs = artifacts.get_aux("step_costs")

    assert len(trajectory) == len(step_costs) == 4
    assert trajectory[0].shape == trajectory[-1].shape == (3, 16, 16)
    assert step_costs[0].shape == step_costs[-1].shape == (16, 16)
    assert artifacts.representation == train_representations[0].to_artifact_dict()


def test_diffusion_inversion_step_costs_reflect_rollout_magnitude() -> None:
    class _ConstantPredictor(nn.Module):
        def forward(self, inputs: torch.Tensor, timesteps: torch.Tensor, noise_scales: torch.Tensor) -> torch.Tensor:
            del timesteps, noise_scales
            return torch.ones_like(inputs)

    representation = make_pixel_output(torch.zeros(3, 16, 16), sample_id="query")
    sample = Sample(image=representation.tensor, sample_id="query")

    small_step = DiffusionInversionBasicNormality(
        input_channels=3,
        hidden_channels=8,
        epochs=1,
        batch_size=1,
        noise_level=0.2,
        num_steps=4,
        step_size=0.1,
    )
    large_step = DiffusionInversionBasicNormality(
        input_channels=3,
        hidden_channels=8,
        epochs=1,
        batch_size=1,
        noise_level=0.2,
        num_steps=4,
        step_size=0.5,
    )
    small_step.denoiser = _ConstantPredictor()
    large_step.denoiser = _ConstantPredictor()

    small_costs = torch.stack(small_step.infer(sample, representation).get_aux("step_costs"))
    large_costs = torch.stack(large_step.infer(sample, representation).get_aux("step_costs"))

    assert not torch.allclose(small_costs, large_costs)
