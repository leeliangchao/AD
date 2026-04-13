"""Process evidence smoke tests for legacy and diffusers inversion backends."""

import torch

from adrf.core.sample import Sample
from adrf.evidence.direction_mismatch import DirectionMismatchEvidence
from adrf.evidence.path_cost import PathCostEvidence
from adrf.normality.diffusion_inversion_basic import DiffusionInversionBasicNormality


def _pixel_representation(image: torch.Tensor) -> dict[str, object]:
    return {
        "representation": image,
        "space_type": "pixel",
        "spatial_shape": tuple(image.shape[-2:]),
    }


def test_process_evidence_reuses_legacy_and_diffusers_inversion_artifacts() -> None:
    """PathCostEvidence and DirectionMismatchEvidence should work on both inversion backends."""

    torch.manual_seed(0)
    train_representations = [
        _pixel_representation(torch.rand(3, 16, 16)),
        _pixel_representation(torch.rand(3, 16, 16)),
    ]
    sample = Sample(image=train_representations[0]["representation"], sample_id="sample-001")

    for backend in ("legacy", "diffusers"):
        model = DiffusionInversionBasicNormality(
            input_channels=3,
            hidden_channels=8,
            learning_rate=1e-3,
            epochs=1,
            batch_size=2,
            noise_level=0.2,
            num_steps=4,
            step_size=0.1,
            backend=backend,
        )
        model.fit(train_representations)
        artifacts = model.infer(sample, train_representations[0])
        path_prediction = PathCostEvidence(aggregator="mean").predict(sample, artifacts)
        direction_prediction = DirectionMismatchEvidence(aggregator="mean", direction_reduce="sum").predict(sample, artifacts)

        assert "anomaly_map" in path_prediction and "image_score" in path_prediction
        assert "anomaly_map" in direction_prediction and "image_score" in direction_prediction
