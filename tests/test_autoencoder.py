"""Tests for the minimal autoencoder normality model."""

from pathlib import Path
import sys

import torch

from adrf.core.sample import Sample
from adrf.normality.autoencoder import AutoEncoderNormality

sys.path.insert(0, str(Path(__file__).parent))

from support.representation_builders import make_pixel_output


def test_autoencoder_fit_and_infer_return_projection_and_reconstruction() -> None:
    """Autoencoder normality should train on pixel tensors and emit artifacts."""

    generator = torch.Generator().manual_seed(0)
    train_representations = [
        make_pixel_output(torch.rand(3, 32, 32, generator=generator), sample_id=f"train-{index:03d}")
        for index in range(4)
    ]

    model = AutoEncoderNormality(
        input_channels=3,
        hidden_channels=8,
        latent_channels=16,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
    )
    model.fit(train_representations)

    sample = Sample(image=train_representations[0].tensor, sample_id="query")
    artifacts = model.infer(sample, train_representations[0])

    assert artifacts.has("projection")
    assert artifacts.has("reconstruction")
    assert artifacts.get_primary("projection").ndim == 3
    assert artifacts.get_primary("reconstruction").shape == (3, 32, 32)
    assert artifacts.representation == train_representations[0].to_artifact_dict()


def test_autoencoder_normalizes_legacy_representation_artifacts() -> None:
    generator = torch.Generator().manual_seed(0)
    train_representations = [
        {
            "representation": torch.rand(3, 16, 16, generator=generator),
            "space_type": "pixel",
            "spatial_shape": (16, 16),
            "feature_dim": 3,
        }
        for _ in range(2)
    ]

    model = AutoEncoderNormality(
        input_channels=3,
        hidden_channels=4,
        latent_channels=8,
        epochs=1,
        batch_size=2,
    )
    model.fit(train_representations)

    sample = Sample(image=train_representations[0]["representation"], sample_id="query")
    artifacts = model.infer(sample, train_representations[0])

    assert artifacts.representation == {
        "tensor": train_representations[0]["representation"],
        "space": "pixel",
        "spatial_shape": (16, 16),
        "feature_dim": 3,
        "sample_id": None,
        "requires_grad": False,
        "device": "cpu",
        "dtype": "torch.float32",
        "provenance": None,
    }
