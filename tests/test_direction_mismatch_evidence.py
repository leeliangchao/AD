"""Tests for the trajectory-based direction mismatch evidence model."""

import pytest
import torch

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample
from adrf.evidence.direction_mismatch import DirectionMismatchEvidence


def test_direction_mismatch_evidence_outputs_map_and_score_from_trajectory() -> None:
    """Direction mismatch should highlight reversals between consecutive trajectory deltas."""

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

    expected_map = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    assert torch.allclose(prediction["anomaly_map"], expected_map)
    assert prediction["image_score"] == pytest.approx(float(expected_map.mean().item()))
    assert prediction["aux_scores"]["num_steps"] == 3
    assert prediction["aux_scores"]["aggregator"] == "mean"


def test_direction_mismatch_evidence_requires_trajectory_capability() -> None:
    """Missing trajectory capability should fail before prediction."""

    artifacts = NormalityArtifacts(auxiliary={"step_costs": [torch.ones(2, 2)]}, capabilities={"step_costs"})

    with pytest.raises(KeyError, match="trajectory"):
        DirectionMismatchEvidence().predict(Sample(image=torch.zeros(1, 2, 2)), artifacts)


def test_direction_mismatch_evidence_distinguishes_monotonic_and_reversing_trajectories() -> None:
    monotonic = [
        torch.tensor([[[0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32),
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32),
        torch.tensor([[[2.0, 0.0], [0.0, 2.0]]], dtype=torch.float32),
    ]
    reversing = [
        torch.tensor([[[0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32),
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32),
        torch.tensor([[[0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32),
    ]

    monotonic_prediction = DirectionMismatchEvidence(aggregator="mean", direction_reduce="sum").predict(
        Sample(image=torch.zeros(1, 2, 2)),
        NormalityArtifacts(auxiliary={"trajectory": monotonic}, capabilities={"trajectory"}),
    )
    reversing_prediction = DirectionMismatchEvidence(aggregator="mean", direction_reduce="sum").predict(
        Sample(image=torch.zeros(1, 2, 2)),
        NormalityArtifacts(auxiliary={"trajectory": reversing}, capabilities={"trajectory"}),
    )

    assert not torch.allclose(monotonic_prediction["anomaly_map"], reversing_prediction["anomaly_map"])
    assert monotonic_prediction["image_score"] != pytest.approx(reversing_prediction["image_score"])


def test_direction_mismatch_evidence_accepts_canonical_step_updates() -> None:
    step_updates = [
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32),
        torch.tensor([[[-1.0, 0.0], [0.0, -1.0]]], dtype=torch.float32),
    ]
    prediction = DirectionMismatchEvidence(aggregator="mean", direction_reduce="sum").predict(
        Sample(image=torch.zeros(1, 2, 2)),
        NormalityArtifacts(auxiliary={"step_updates": step_updates}, capabilities={"step_updates"}),
    )

    assert prediction["anomaly_map"].shape == (2, 2)
    assert isinstance(prediction["image_score"], float)
