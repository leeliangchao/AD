"""Tests for the minimal diffusion-style normality model."""

from collections.abc import Iterable
from pathlib import Path
import sys
from typing import get_type_hints

import torch

from adrf.core.sample import Sample
from adrf.normality.diffusion_basic import DiffusionBasicNormality
from adrf.representation.contracts import RepresentationOutput

sys.path.insert(0, str(Path(__file__).parent))

from support.representation_builders import make_pixel_output


def test_diffusion_basic_public_annotations_expose_representation_output_only() -> None:
    fit_hints = get_type_hints(DiffusionBasicNormality.fit)
    infer_hints = get_type_hints(DiffusionBasicNormality.infer)

    assert fit_hints["representations"] == Iterable[RepresentationOutput]
    assert infer_hints["representation"] is RepresentationOutput


def test_diffusion_basic_fit_and_infer_accept_representation_output() -> None:
    """DiffusionBasicNormality should train on typed pixel outputs and emit normalized artifacts."""

    generator = torch.Generator().manual_seed(0)
    representations = [
        make_pixel_output(torch.rand(3, 16, 16, generator=generator), sample_id=f"train-{index:03d}")
        for index in range(2)
    ]
    model = DiffusionBasicNormality(
        input_channels=3,
        hidden_channels=8,
        time_embed_dim=32,
        num_train_timesteps=32,
        learning_rate=1e-3,
        epochs=1,
        batch_size=2,
        noise_level=0.2,
    )

    model.fit(representations)
    sample = Sample(image=representations[0].tensor, sample_id="query")
    artifacts = model.infer(sample, representations[0])

    assert artifacts.has("predicted_noise")
    assert artifacts.has("target_noise")
    assert "anomaly_map" not in artifacts.primary
    predicted_noise = artifacts.get_aux("predicted_noise")
    target_noise = artifacts.get_aux("target_noise")
    assert isinstance(predicted_noise, torch.Tensor)
    assert isinstance(target_noise, torch.Tensor)
    assert predicted_noise.shape == target_noise.shape == (3, 16, 16)
    assert artifacts.get_diag("time_embed_dim") == 32
    assert artifacts.get_diag("num_train_timesteps") == 32
    assert artifacts.representation == representations[0].to_artifact_dict()
    assert artifacts.representation["space"] == "pixel"
    assert artifacts.representation["provenance"]["representation_name"] == "pixel"
    assert DiffusionBasicNormality.accepted_spaces == frozenset({"pixel"})
    assert DiffusionBasicNormality.accepted_tensor_ranks == frozenset({3})
