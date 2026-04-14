"""Tests for the feature memory normality model."""

from pathlib import Path
import sys

import torch

from adrf.core.sample import Sample
from adrf.normality.feature_memory import FeatureMemoryNormality

sys.path.insert(0, str(Path(__file__).parent))

from support.representation_builders import make_feature_output


def test_feature_memory_fit_and_infer_return_expected_artifacts() -> None:
    """Feature memory should build a memory bank and emit standardized artifacts."""

    representation = make_feature_output(torch.ones(4, 2, 2), sample_id="train-001")
    samples = [Sample(image=torch.zeros(3, 8, 8), label=0) for _ in range(2)]

    model = FeatureMemoryNormality()
    model.fit([representation, representation], samples)

    artifacts = model.infer(Sample(image=torch.zeros(3, 8, 8), sample_id="query"), representation)

    assert artifacts.has("normality_embedding")
    assert artifacts.has("feature_response")
    assert artifacts.has("memory_distance")
    assert artifacts.get_primary("normality_embedding").shape == (4,)
    assert artifacts.get_aux("feature_response").shape == (4, 2, 2)
    assert artifacts.get_aux("memory_distance").shape == (2, 2)
    assert torch.allclose(artifacts.get_aux("memory_distance"), torch.zeros(2, 2))
    assert artifacts.representation == representation.to_artifact_dict()


def test_feature_memory_normalizes_legacy_representation_artifacts() -> None:
    representation = {
        "representation": torch.ones(4, 2, 2),
        "space_type": "feature",
        "spatial_shape": (2, 2),
        "feature_dim": 4,
    }

    model = FeatureMemoryNormality()
    model.fit([representation])

    artifacts = model.infer(Sample(image=torch.zeros(3, 8, 8), sample_id="query"), representation)

    assert artifacts.representation == {
        "tensor": representation["representation"],
        "space": "feature",
        "spatial_shape": (2, 2),
        "feature_dim": 4,
        "sample_id": None,
        "requires_grad": False,
        "device": "cpu",
        "dtype": "torch.float32",
        "provenance": None,
    }
