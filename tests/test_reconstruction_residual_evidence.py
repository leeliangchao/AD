"""Tests for reconstruction-residual evidence generation."""

import pytest
import torch

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample
from adrf.evidence.reconstruction_residual import ReconstructionResidualEvidence


def test_reconstruction_residual_evidence_outputs_l1_map_and_score() -> None:
    """Residual evidence should compare the sample image with the reconstruction."""

    sample = Sample(image=torch.tensor([[[0.0, 1.0], [0.5, 0.0]]], dtype=torch.float32))
    artifacts = NormalityArtifacts(
        primary={
            "reconstruction": torch.tensor([[[0.0, 0.0], [0.25, 0.25]]], dtype=torch.float32),
        },
        capabilities={"reconstruction"},
    )

    prediction = ReconstructionResidualEvidence(aggregator="mean").predict(sample, artifacts)

    expected_map = torch.tensor([[0.0, 1.0], [0.25, 0.25]], dtype=torch.float32)
    assert torch.allclose(prediction["anomaly_map"], expected_map)
    assert prediction["image_score"] == pytest.approx(float(expected_map.mean().item()))
    assert prediction["aux_scores"]["aggregator"] == "mean"


def test_reconstruction_residual_evidence_accepts_projection_fallback() -> None:
    """Projection should be accepted as a fallback capability when reconstruction is absent."""

    sample = Sample(image=torch.zeros(1, 2, 2))
    artifacts = NormalityArtifacts(
        primary={"projection": torch.zeros(1, 2, 2)},
        capabilities={"projection"},
    )

    prediction = ReconstructionResidualEvidence().predict(sample, artifacts)

    assert prediction["image_score"] == pytest.approx(0.0)

