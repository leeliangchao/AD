"""Tests for feature-distance evidence generation."""

import pytest
import torch

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample
from adrf.evidence.feature_distance import FeatureDistanceEvidence


def test_feature_distance_evidence_outputs_map_and_score() -> None:
    """Feature distance evidence should convert memory distances into unified predictions."""

    artifacts = NormalityArtifacts(
        auxiliary={
            "feature_response": torch.ones(4, 2, 2),
            "memory_distance": torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32),
        },
        capabilities={"feature_response", "memory_distance"},
    )
    sample = Sample(image=torch.zeros(3, 8, 8), sample_id="sample-001")

    prediction = FeatureDistanceEvidence(aggregator="max").predict(sample, artifacts)

    assert prediction["anomaly_map"].shape == (2, 2)
    assert prediction["image_score"] == pytest.approx(0.4)
    assert prediction["aux_scores"]["aggregator"] == "max"


def test_feature_distance_evidence_requires_declared_capabilities() -> None:
    """Missing artifact capabilities should raise before prediction."""

    artifacts = NormalityArtifacts(capabilities={"feature_response"})
    sample = Sample(image=torch.zeros(3, 8, 8))

    with pytest.raises(KeyError, match="memory_distance"):
        FeatureDistanceEvidence().predict(sample, artifacts)

