"""Tests for DiffusionInversionBasicNormality with the diffusers backend."""

from pathlib import Path
import sys

import torch

from adrf.core.sample import Sample
from adrf.normality.diffusion_inversion_basic import DiffusionInversionBasicNormality

sys.path.insert(0, str(Path(__file__).parent))

from support.representation_builders import make_pixel_output


def test_diffusion_inversion_diffusers_backend_fit_and_infer_emit_process_artifacts() -> None:
    """The diffusers backend should expose the same process artifact contract."""

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

    assert artifacts.has("trajectory")
    assert artifacts.has("step_costs")
    assert isinstance(artifacts.get_aux("trajectory"), list)
    assert isinstance(artifacts.get_aux("step_costs"), list)
    assert artifacts.representation == train_representations[0].to_artifact_dict()
