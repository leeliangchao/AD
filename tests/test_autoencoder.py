"""Tests for the minimal autoencoder normality model."""

import torch

from adrf.core.sample import Sample
from adrf.normality.autoencoder import AutoEncoderNormality


def _pixel_representation(image: torch.Tensor) -> dict[str, object]:
    return {
        "representation": image,
        "space_type": "pixel",
        "spatial_shape": tuple(image.shape[-2:]),
    }


def test_autoencoder_fit_and_infer_return_projection_and_reconstruction() -> None:
    """Autoencoder normality should train on pixel tensors and emit artifacts."""

    generator = torch.Generator().manual_seed(0)
    train_representations = [
        _pixel_representation(torch.rand(3, 32, 32, generator=generator))
        for _ in range(4)
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

    sample = Sample(image=train_representations[0]["representation"], sample_id="query")
    artifacts = model.infer(sample, train_representations[0])

    assert artifacts.has("projection")
    assert artifacts.has("reconstruction")
    assert artifacts.get_primary("projection").ndim == 3
    assert artifacts.get_primary("reconstruction").shape == (3, 32, 32)

