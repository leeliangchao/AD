"""Tests for shared diffusion core helpers."""

from __future__ import annotations

import torch

from adrf.normality.diffusion_core import sample_legacy_noisy_inputs, sample_legacy_timesteps


def test_sample_legacy_timesteps_uses_max_timestep_for_inference() -> None:
    timesteps = sample_legacy_timesteps(
        batch_size=3,
        device=torch.device("cpu"),
        num_train_timesteps=32,
        inference=True,
    )

    assert torch.equal(timesteps, torch.full((3,), 31, dtype=torch.long))


def test_sample_legacy_noisy_inputs_preserves_provided_noise_and_reports_scales() -> None:
    clean_batch = torch.zeros(2, 3, 4, 4)
    target_noise = torch.ones_like(clean_batch)

    noisy_batch, returned_noise, timesteps, noise_scales = sample_legacy_noisy_inputs(
        clean_batch,
        num_train_timesteps=16,
        noise_level=0.2,
        inference=True,
        target_noise=target_noise,
    )

    assert torch.equal(returned_noise, target_noise)
    assert torch.equal(timesteps, torch.full((2,), 15, dtype=torch.long))
    assert noise_scales.shape == (2,)
    assert torch.allclose(noisy_batch, noise_scales.view(-1, 1, 1, 1))
