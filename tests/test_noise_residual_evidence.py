"""Tests for the diffusion noise-residual evidence model."""

import pytest
import torch

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample
from adrf.evidence.noise_residual import NoiseResidualEvidence


def test_noise_residual_evidence_computes_channel_reduced_map_and_score() -> None:
    """Noise residual evidence should reduce channel residuals into a 2D anomaly map."""

    predicted_noise = torch.tensor(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    target_noise = torch.zeros_like(predicted_noise)
    artifacts = NormalityArtifacts(
        auxiliary={"predicted_noise": predicted_noise, "target_noise": target_noise},
        capabilities={"predicted_noise", "target_noise"},
    )

    prediction = NoiseResidualEvidence(aggregator="mean").predict(Sample(image=torch.zeros(2, 2, 2)), artifacts)

    expected_map = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
    assert torch.allclose(prediction["anomaly_map"], expected_map)
    assert prediction["image_score"] == pytest.approx(0.5)
    assert prediction["aux_scores"]["aggregator"] == "mean"


def test_noise_residual_evidence_requires_noise_capabilities() -> None:
    """Missing noise artifacts should raise before prediction."""

    artifacts = NormalityArtifacts(
        auxiliary={"predicted_noise": torch.zeros(1, 2, 2)},
        capabilities={"predicted_noise"},
    )

    with pytest.raises(KeyError, match="target_noise"):
        NoiseResidualEvidence().predict(Sample(image=torch.zeros(1, 2, 2)), artifacts)

