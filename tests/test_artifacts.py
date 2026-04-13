"""Tests for the normality artifacts exchange layer."""

import pytest

from adrf.core.artifacts import NormalityArtifacts


def test_artifacts_accessors_and_capability_checks() -> None:
    """Artifacts should expose typed accessors and capability validation."""

    artifacts = NormalityArtifacts(
        context={"sample_id": "sample-001"},
        representation={"space_type": "feature"},
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


def test_artifacts_require_raises_for_missing_capabilities() -> None:
    """Missing capabilities should fail fast with a useful error."""

    artifacts = NormalityArtifacts(capabilities={"projection"})

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

