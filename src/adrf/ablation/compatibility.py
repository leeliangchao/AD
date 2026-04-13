"""Compatibility rules for ablation matrix combinations."""

from __future__ import annotations

from typing import Any


def explain_incompatibility(combo: dict[str, Any], rules: dict[str, Any]) -> list[str]:
    """Return human-readable reasons explaining why a combo is invalid."""

    reasons: list[str] = []

    normality = combo.get("normality")
    evidence = combo.get("evidence")
    representation = combo.get("representation")
    dataset = combo.get("dataset")
    protocol = combo.get("protocol")

    normality_evidence = rules.get("normality_evidence", {})
    allowed_evidence = normality_evidence.get(normality)
    if allowed_evidence is not None and evidence not in allowed_evidence:
        reasons.append(
            f"evidence `{evidence}` is not compatible with normality `{normality}`."
        )

    representation_normality = rules.get("representation_normality", {})
    allowed_representations = representation_normality.get(normality)
    if allowed_representations is not None and representation not in allowed_representations:
        reasons.append(
            f"representation `{representation}` is not compatible with normality `{normality}`."
        )

    dataset_protocol = rules.get("dataset_protocol", {})
    allowed_protocols = dataset_protocol.get(dataset)
    if allowed_protocols is not None and protocol not in allowed_protocols:
        reasons.append(
            f"protocol `{protocol}` is not compatible with dataset `{dataset}`."
        )

    return reasons


def is_compatible(combo: dict[str, Any], rules: dict[str, Any]) -> bool:
    """Return whether a combo satisfies all configured compatibility rules."""

    return not explain_incompatibility(combo, rules)


def filter_valid_combinations(
    combos: list[dict[str, Any]],
    rules: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split candidate combinations into valid and invalid subsets."""

    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for combo in combos:
        reasons = explain_incompatibility(combo, rules)
        if reasons:
            invalid.append({"combo": combo, "reasons": reasons})
        else:
            valid.append(combo)
    return valid, invalid

