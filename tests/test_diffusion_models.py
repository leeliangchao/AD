"""Tests for reusable diffusion model backbones."""

from __future__ import annotations

import torch

from adrf.normality.diffusion_models import ConditionedNoisePredictor, NoisePredictor


def test_noise_predictor_supports_optional_class_conditioning() -> None:
    model = NoisePredictor(
        input_channels=3,
        base_channels=8,
        channel_mults=(1, 2),
        num_res_blocks=2,
        time_embed_dim=32,
        conditioning_hidden_dim=64,
        num_classes=3,
        class_embed_dim=8,
    )
    inputs = torch.randn(2, 3, 16, 16)
    timesteps = torch.tensor([1, 2], dtype=torch.long)
    noise_scales = torch.tensor([0.1, 0.2], dtype=torch.float32)
    class_ids = torch.tensor([0, 2], dtype=torch.long)

    output = model(inputs, timesteps, noise_scales, class_ids=class_ids)

    assert output.shape == inputs.shape


def test_noise_predictor_class_conditioning_changes_output_when_embeddings_differ() -> None:
    model = NoisePredictor(
        input_channels=3,
        base_channels=8,
        channel_mults=(1,),
        num_res_blocks=1,
        time_embed_dim=8,
        conditioning_hidden_dim=16,
        num_classes=2,
        class_embed_dim=4,
    )
    with torch.no_grad():
        model.class_embedding.weight[0].fill_(0.0)
        model.class_embedding.weight[1].fill_(1.0)

    inputs = torch.zeros(1, 3, 8, 8)
    timesteps = torch.tensor([1], dtype=torch.long)
    noise_scales = torch.tensor([0.1], dtype=torch.float32)

    zero_output = model(inputs, timesteps, noise_scales, class_ids=torch.tensor([0], dtype=torch.long))
    one_output = model(inputs, timesteps, noise_scales, class_ids=torch.tensor([1], dtype=torch.long))

    assert not torch.allclose(zero_output, one_output)


def test_conditioned_noise_predictor_supports_reference_and_class_conditioning() -> None:
    model = ConditionedNoisePredictor(
        input_channels=3,
        base_channels=8,
        condition_channels=6,
        channel_mults=(1, 2),
        num_res_blocks=2,
        time_embed_dim=16,
        num_classes=4,
        class_embed_dim=8,
    )
    inputs = torch.randn(2, 6, 16, 16)
    timesteps = torch.tensor([1, 2], dtype=torch.long)
    class_ids = torch.tensor([1, 3], dtype=torch.long)

    output = model(inputs, timesteps, class_ids=class_ids)

    assert output.shape == (2, 3, 16, 16)
