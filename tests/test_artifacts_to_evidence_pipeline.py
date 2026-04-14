"""Smoke tests for the artifacts-to-evidence-to-evaluator flow."""

from pathlib import Path
import sys

import pytest
import torch

from adrf.core.sample import Sample
from adrf.evaluation.evaluator import BasicADEvaluator
from adrf.evidence.feature_distance import FeatureDistanceEvidence
from adrf.evidence.reconstruction_residual import ReconstructionResidualEvidence
from adrf.normality.autoencoder import AutoEncoderNormality
from adrf.normality.feature_memory import FeatureMemoryNormality

sys.path.insert(0, str(Path(__file__).parent))

from support.representation_builders import make_feature_output, make_pixel_output


def test_feature_memory_artifacts_flow_into_evidence_and_evaluator() -> None:
    """Feature-memory artifacts should be consumable by evidence and evaluator."""

    train_representation = make_feature_output(torch.ones(4, 2, 2), sample_id="train-001")
    normal_query_representation = make_feature_output(torch.ones(4, 2, 2), sample_id="sample-000")
    anomaly_query_representation = make_feature_output(torch.full((4, 2, 2), 2.0), sample_id="sample-001")
    normality = FeatureMemoryNormality()
    normality.fit([train_representation])
    normal_sample = Sample(
        image=torch.zeros(3, 8, 8),
        label=0,
        mask=torch.zeros(1, 2, 2),
        sample_id="sample-000",
    )
    anomaly_sample = Sample(
        image=torch.zeros(3, 8, 8),
        label=1,
        mask=torch.ones(1, 2, 2),
        sample_id="sample-001",
    )

    normal_prediction = FeatureDistanceEvidence(aggregator="max").predict(
        normal_sample,
        normality.infer(normal_sample, normal_query_representation),
    )
    anomaly_prediction = FeatureDistanceEvidence(aggregator="max").predict(
        anomaly_sample,
        normality.infer(anomaly_sample, anomaly_query_representation),
    )

    assert set(normal_prediction) == {"anomaly_map", "image_score", "aux_scores"}
    assert set(anomaly_prediction) == {"anomaly_map", "image_score", "aux_scores"}

    evaluator = BasicADEvaluator()
    evaluator.update(normal_prediction, normal_sample)
    evaluator.update(anomaly_prediction, anomaly_sample)

    metrics = evaluator.compute()

    assert metrics["image_auroc"] == pytest.approx(1.0)
    assert metrics["pixel_auroc"] == pytest.approx(1.0)
    assert metrics["pixel_aupr"] == pytest.approx(1.0)


def test_autoencoder_artifacts_flow_into_residual_evidence() -> None:
    """Autoencoder artifacts should be consumable by reconstruction evidence."""

    generator = torch.Generator().manual_seed(0)
    train_representations = [
        make_pixel_output(torch.rand(3, 16, 16, generator=generator), sample_id=f"train-{index:03d}")
        for index in range(2)
    ]
    normality = AutoEncoderNormality(
        input_channels=3,
        hidden_channels=4,
        latent_channels=8,
        epochs=1,
        batch_size=2,
    )
    normality.fit(train_representations)

    sample = Sample(
        image=train_representations[0].tensor,
        label=0,
        mask=torch.zeros(1, 16, 16),
        sample_id="sample-002",
    )
    artifacts = normality.infer(sample, train_representations[0])
    prediction = ReconstructionResidualEvidence(aggregator="mean").predict(sample, artifacts)

    assert prediction["anomaly_map"].shape == (16, 16)
    assert isinstance(prediction["image_score"], float)
    assert prediction["aux_scores"]["aggregator"] == "mean"
