"""Tests for the diffusers scheduler adapter."""

import torch

from adrf.diffusion.schedulers import make_scheduler


def test_diffusers_scheduler_adapter_add_noise_and_step_preserve_shape() -> None:
    """The scheduler adapter should add noise and step with stable tensor shapes."""

    scheduler = make_scheduler("ddpm", num_train_timesteps=10)
    sample = torch.rand(2, 3, 16, 16)
    noise = torch.randn_like(sample)
    timesteps = torch.tensor([1, 5], dtype=torch.long)

    noisy = scheduler.add_noise(sample, noise, timesteps)
    step_result = scheduler.step(noise, int(timesteps[0].item()), noisy)

    assert noisy.shape == sample.shape
    assert step_result.shape == sample.shape

