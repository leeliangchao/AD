"""Tests for the trajectory-based direction mismatch evidence model."""

import pytest
import torch

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample
from adrf.evidence.direction_mismatch import DirectionMismatchEvidence


def test_direction_mismatch_evidence_outputs_map_and_score_from_trajectory() -> None:
    """Direction mismatch should use only trajectory step differences."""

    trajectory = [
        torch.tensor([[[0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32),
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32),
        torch.tensor([[[1.0, 1.0], [0.0, 0.0]]], dtype=torch.float32),
    ]
    artifacts = NormalityArtifacts(
        auxiliary={"trajectory": trajectory, "step_costs": [torch.ones(2, 2)]},
        capabilities={"trajectory", "step_costs"},
    )

    prediction = DirectionMismatchEvidence(aggregator="mean", direction_reduce="sum").predict(
        Sample(image=torch.zeros(1, 2, 2)),
        artifacts,
    )

    expected_map = torch.tensor([[1.0, 1.0], [0.0, 2.0]], dtype=torch.float32)
    assert torch.allclose(prediction["anomaly_map"], expected_map)
    assert prediction["image_score"] == pytest.approx(float(expected_map.mean().item()))
    assert prediction["aux_scores"]["num_steps"] == 3
    assert prediction["aux_scores"]["aggregator"] == "mean"


def test_direction_mismatch_evidence_requires_trajectory_capability() -> None:
    """Missing trajectory capability should fail before prediction."""

    artifacts = NormalityArtifacts(auxiliary={"step_costs": [torch.ones(2, 2)]}, capabilities={"step_costs"})

    with pytest.raises(KeyError, match="trajectory"):
        DirectionMismatchEvidence().predict(Sample(image=torch.zeros(1, 2, 2)), artifacts)

