"""Tests for DiffusionBasicNormality with the diffusers backend."""

from pathlib import Path
import sys

import torch

from adrf.core.sample import Sample
from adrf.normality.diffusion_basic import DiffusionBasicNormality

sys.path.insert(0, str(Path(__file__).parent))

from support.representation_builders import make_pixel_output


def test_diffusion_basic_diffusers_backend_fit_and_infer_emit_noise_artifacts() -> None:
    """The diffusers backend should train and infer with the same artifact schema."""

    generator = torch.Generator().manual_seed(0)
    train_representations = [
        make_pixel_output(torch.rand(3, 16, 16, generator=generator), sample_id=f"train-{index:03d}")
        for index in range(2)
    ]
    model = DiffusionBasicNormality(
        input_channels=3,
        hidden_channels=8,
        learning_rate=1e-3,
        epochs=1,
        batch_size=2,
        noise_level=0.2,
        backend="diffusers",
    )

    model.fit(train_representations)
    sample = Sample(image=train_representations[0].tensor, sample_id="query")
    artifacts = model.infer(sample, train_representations[0])

    assert artifacts.has("predicted_noise")
    assert artifacts.has("target_noise")
    predicted_noise = artifacts.get_aux("predicted_noise")
    target_noise = artifacts.get_aux("target_noise")
    reconstruction = artifacts.get_primary("reconstruction")
    assert predicted_noise.shape == target_noise.shape == (3, 16, 16)
    assert reconstruction.shape == (3, 16, 16)
    assert artifacts.representation == train_representations[0].to_artifact_dict()


def test_diffusion_basic_diffusers_backend_respects_num_train_timesteps() -> None:
    generator = torch.Generator().manual_seed(0)
    train_representations = [
        make_pixel_output(torch.rand(3, 16, 16, generator=generator), sample_id="train-000")
    ]
    model = DiffusionBasicNormality(
        input_channels=3,
        hidden_channels=8,
        learning_rate=1e-3,
        epochs=1,
        batch_size=1,
        noise_level=0.2,
        num_train_timesteps=32,
        backend="diffusers",
    )

    model.fit(train_representations)

    assert model.diffusers_adapter is not None
    assert model.diffusers_adapter.scheduler.scheduler.config.num_train_timesteps == 32


def test_diffusion_basic_diffusers_backend_infer_does_not_sample_random_timestep(monkeypatch) -> None:
    generator = torch.Generator().manual_seed(0)
    train_representations = [
        make_pixel_output(torch.rand(3, 16, 16, generator=generator), sample_id="train-000")
    ]
    model = DiffusionBasicNormality(
        input_channels=3,
        hidden_channels=8,
        learning_rate=1e-3,
        epochs=1,
        batch_size=1,
        noise_level=0.2,
        num_train_timesteps=32,
        backend="diffusers",
    )
    model.fit(train_representations)
    sample = Sample(image=train_representations[0].tensor, sample_id="query")

    def _forbid_randint(*args, **kwargs):
        raise RuntimeError("randint called")

    monkeypatch.setattr("adrf.diffusion.adapters.torch.randint", _forbid_randint)

    artifacts = model.infer(sample, train_representations[0])

    assert artifacts.has("predicted_noise")


def test_diffusion_basic_diffusers_backend_supports_optional_class_conditioning() -> None:
    generator = torch.Generator().manual_seed(0)
    samples = [
        Sample(image=torch.rand(3, 16, 16, generator=generator), sample_id="train-000", category="bottle"),
        Sample(image=torch.rand(3, 16, 16, generator=generator), sample_id="train-001", category="capsule"),
    ]
    train_representations = [make_pixel_output(sample.image, sample_id=sample.sample_id or "sample") for sample in samples]
    model = DiffusionBasicNormality(
        input_channels=3,
        hidden_channels=8,
        learning_rate=1e-3,
        epochs=1,
        batch_size=2,
        noise_level=0.2,
        num_train_timesteps=32,
        backend="diffusers",
        num_classes=2,
    )

    model.fit(train_representations, samples)
    artifacts = model.infer(samples[0], train_representations[0])

    assert model.class_to_index == {"bottle": 0, "capsule": 1}
    assert artifacts.get_primary("reconstruction").shape == (3, 16, 16)
