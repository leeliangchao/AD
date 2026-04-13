"""Tests for ablation compatibility filtering."""

from adrf.ablation.compatibility import explain_incompatibility, filter_valid_combinations, is_compatible


def test_ablation_compatibility_filters_invalid_pairs() -> None:
    """Compatibility rules should remove invalid normality/evidence combinations."""

    combos = [
        {"normality": "diffusion_basic", "evidence": "noise_residual", "representation": "pixel"},
        {"normality": "diffusion_basic", "evidence": "path_cost", "representation": "pixel"},
        {"normality": "feature_memory", "evidence": "feature_distance", "representation": "feature"},
    ]
    rules = {
        "normality_evidence": {
            "diffusion_basic": ["noise_residual"],
            "feature_memory": ["feature_distance"],
        }
    }

    valid, invalid = filter_valid_combinations(combos, rules)

    assert len(valid) == 2
    assert len(invalid) == 1
    assert invalid[0]["combo"]["evidence"] == "path_cost"
    assert "not compatible" in invalid[0]["reasons"][0]


def test_ablation_compatibility_checks_representation_rules() -> None:
    """Representation compatibility should also be enforceable when configured."""

    combo = {"normality": "feature_memory", "evidence": "feature_distance", "representation": "pixel"}
    rules = {
        "normality_evidence": {"feature_memory": ["feature_distance"]},
        "representation_normality": {"feature_memory": ["feature"]},
    }

    assert is_compatible(combo, rules) is False
    reasons = explain_incompatibility(combo, rules)
    assert any("representation" in reason for reason in reasons)

