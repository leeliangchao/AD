"""Tests for the normality artifacts exchange layer."""

from pathlib import Path
import sys

import pytest
import torch

from adrf.core.artifacts import NormalityArtifacts

sys.path.insert(0, str(Path(__file__).parent))

from support.representation_builders import make_feature_output


def test_artifacts_accessors_and_capability_checks() -> None:
    """Artifacts should expose typed accessors and capability validation."""

    representation = make_feature_output(torch.ones(4), sample_id="sample-001")
    artifacts = NormalityArtifacts(
        context={"sample_id": "sample-001"},
        representation=representation.to_artifact_dict(),
        primary={"normality_embedding": [1.0, 2.0]},
        auxiliary={"memory_distance": 0.25},
        diagnostics={"uncertainty": 0.1},
        capabilities={"normality_embedding", "memory_distance", "uncertainty"},
    )

    assert artifacts.has("memory_distance") is True
    assert artifacts.has("projection") is False
    artifacts.require("normality_embedding", "memory_distance")
    assert artifacts.get_primary("normality_embedding") == [1.0, 2.0]
    assert artifacts.get_aux("memory_distance") == 0.25
    assert artifacts.get_aux("missing", "fallback") == "fallback"
    assert artifacts.get_diag("uncertainty") == 0.1
    assert artifacts.representation == representation.to_artifact_dict()


def test_artifacts_require_raises_for_missing_capabilities() -> None:
    """Missing capabilities should fail fast with a useful error."""

    artifacts = NormalityArtifacts(primary={"projection": torch.zeros(1)}, capabilities={"projection"})

    with pytest.raises(KeyError, match="reconstruction"):
        artifacts.require("projection", "reconstruction")


def test_artifacts_use_independent_default_containers() -> None:
    """Default dictionaries and sets should not be shared across instances."""

    first = NormalityArtifacts()
    second = NormalityArtifacts()

    first.context["state"] = "tracked"
    first.capabilities.add("projection")

    assert second.context == {}
    assert second.capabilities == set()


def test_artifacts_round_trip_serialized_payload() -> None:
    artifacts = NormalityArtifacts(
        context={"sample_id": "sample-001"},
        representation={"space": "feature"},
        primary={"projection": [1.0]},
        auxiliary={"memory_distance": 0.25},
        diagnostics={"fit_loss": 0.1},
        capabilities={"projection", "memory_distance"},
    )

    restored = NormalityArtifacts.from_mapping(artifacts.to_dict())

    assert restored.context == {"sample_id": "sample-001"}
    assert restored.representation == {"space": "feature"}
    assert restored.primary == {"projection": [1.0]}
    assert restored.auxiliary == {"memory_distance": 0.25}
    assert restored.diagnostics == {"fit_loss": 0.1}
    assert restored.capabilities == {"projection", "memory_distance"}


def test_artifacts_validate_rejects_capabilities_missing_from_payloads() -> None:
    artifacts = NormalityArtifacts(
        primary={"projection": [1.0]},
        capabilities={"projection", "reconstruction"},
    )

    with pytest.raises(ValueError, match="reconstruction"):
        artifacts.validate()
