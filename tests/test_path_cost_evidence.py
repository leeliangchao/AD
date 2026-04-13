"""Tests for the path-cost evidence model."""

import pytest
import torch

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample
from adrf.evidence.path_cost import PathCostEvidence


def test_path_cost_evidence_sums_step_cost_maps_into_prediction() -> None:
    """PathCostEvidence should accumulate step costs into a 2D anomaly map."""

    step_costs = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32),
    ]
    artifacts = NormalityArtifacts(
        auxiliary={"step_costs": step_costs},
        capabilities={"step_costs"},
    )

    prediction = PathCostEvidence(aggregator="mean").predict(Sample(image=torch.zeros(1, 2, 2)), artifacts)

    expected_map = torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float32)
    assert torch.allclose(prediction["anomaly_map"], expected_map)
    assert prediction["image_score"] == pytest.approx(float(expected_map.mean().item()))
    assert prediction["aux_scores"]["path_cost_total"] == pytest.approx(float(expected_map.sum().item()))


def test_path_cost_evidence_requires_step_cost_capability() -> None:
    """Missing step_costs capability should fail before prediction."""

    artifacts = NormalityArtifacts(auxiliary={}, capabilities=set())

    with pytest.raises(KeyError, match="step_costs"):
        PathCostEvidence().predict(Sample(image=torch.zeros(1, 2, 2)), artifacts)

