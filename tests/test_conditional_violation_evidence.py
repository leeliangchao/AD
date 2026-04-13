"""Tests for reference-conditioned violation evidence."""

import pytest
import torch

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample
from adrf.evidence.conditional_violation import ConditionalViolationEvidence


def test_conditional_violation_evidence_outputs_map_and_score() -> None:
    """ConditionalViolationEvidence should convert projection deviation into anomaly evidence."""

    sample = Sample(
        image=torch.tensor([[[0.0, 1.0], [0.5, 0.0]]], dtype=torch.float32),
        reference=torch.zeros(1, 2, 2),
    )
    artifacts = NormalityArtifacts(
        primary={
            "reference_projection": torch.tensor([[[0.0, 0.0], [0.25, 0.25]]], dtype=torch.float32),
        },
        capabilities={"reference_projection"},
    )

    prediction = ConditionalViolationEvidence(aggregator="mean").predict(sample, artifacts)

    expected_map = torch.tensor([[0.0, 1.0], [0.25, 0.25]], dtype=torch.float32)
    assert torch.allclose(prediction["anomaly_map"], expected_map)
    assert prediction["image_score"] == pytest.approx(float(expected_map.mean().item()))
    assert prediction["aux_scores"]["aggregator"] == "mean"


def test_conditional_violation_evidence_requires_projection_capability() -> None:
    """Missing conditional projection should fail before prediction."""

    sample = Sample(image=torch.zeros(1, 2, 2), reference=torch.zeros(1, 2, 2))
    artifacts = NormalityArtifacts(capabilities=set())

    with pytest.raises(KeyError, match="reference_projection"):
        ConditionalViolationEvidence().predict(sample, artifacts)

