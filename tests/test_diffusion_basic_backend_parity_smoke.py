"""Smoke tests for legacy and diffusers diffusion backends."""

from pathlib import Path
import sys

import torch

from adrf.core.sample import Sample
from adrf.normality.diffusion_basic import DiffusionBasicNormality

sys.path.insert(0, str(Path(__file__).parent))

from support.representation_builders import make_pixel_output


def test_diffusion_basic_legacy_and_diffusers_backends_share_artifact_contract() -> None:
    """Both backends should emit the same artifact keys and tensor shapes."""

    generator = torch.Generator().manual_seed(0)
    train_representations = [
        make_pixel_output(torch.rand(3, 16, 16, generator=generator), sample_id=f"train-{index:03d}")
        for index in range(2)
    ]
    sample = Sample(image=train_representations[0].tensor, sample_id="query")

    legacy = DiffusionBasicNormality(
        input_channels=3,
        hidden_channels=8,
        learning_rate=1e-3,
        epochs=1,
        batch_size=2,
        noise_level=0.2,
        backend="legacy",
    )
    legacy.fit(train_representations)
    legacy_artifacts = legacy.infer(sample, train_representations[0])

    diffusers = DiffusionBasicNormality(
        input_channels=3,
        hidden_channels=8,
        learning_rate=1e-3,
        epochs=1,
        batch_size=2,
        noise_level=0.2,
        backend="diffusers",
    )
    diffusers.fit(train_representations)
    diffusers_artifacts = diffusers.infer(sample, train_representations[0])

    for artifacts in (legacy_artifacts, diffusers_artifacts):
        assert artifacts.has("predicted_noise")
        assert artifacts.has("target_noise")

    assert legacy_artifacts.get_aux("predicted_noise").shape == diffusers_artifacts.get_aux("predicted_noise").shape
    assert legacy_artifacts.get_aux("target_noise").shape == diffusers_artifacts.get_aux("target_noise").shape
    assert legacy_artifacts.representation == train_representations[0].to_artifact_dict()
    assert diffusers_artifacts.representation == train_representations[0].to_artifact_dict()
