"""Internal diffusion task helpers for canonical artifact construction."""

from __future__ import annotations

from typing import Any

import torch

from adrf.core.artifacts import NormalityArtifacts


def build_reconstruction_artifacts(
    *,
    sample_id: str | None,
    category: str | None,
    representation: dict[str, object],
    reconstruction: torch.Tensor,
    predicted_noise: torch.Tensor,
    target_noise: torch.Tensor,
    diagnostics: dict[str, Any],
) -> NormalityArtifacts:
    return NormalityArtifacts(
        context={
            "sample_id": sample_id,
            "category": category,
            "mode": "inference",
        },
        representation=representation,
        primary={"reconstruction": reconstruction},
        auxiliary={
            "predicted_noise": predicted_noise,
            "target_noise": target_noise,
        },
        diagnostics=dict(diagnostics),
        capabilities={"predicted_noise", "target_noise"},
    )


def build_conditional_reconstruction_artifacts(
    *,
    sample_id: str | None,
    category: str | None,
    representation: dict[str, object],
    reconstruction: torch.Tensor,
    reference_projection: torch.Tensor,
    predicted_noise: torch.Tensor,
    target_noise: torch.Tensor,
    conditional_alignment: torch.Tensor,
    diagnostics: dict[str, Any],
) -> NormalityArtifacts:
    return NormalityArtifacts(
        context={
            "sample_id": sample_id,
            "category": category,
            "has_reference": True,
            "mode": "inference",
        },
        representation=representation,
        primary={
            "reconstruction": reconstruction,
            "reference_projection": reference_projection,
        },
        auxiliary={
            "predicted_noise": predicted_noise,
            "target_noise": target_noise,
            "conditional_alignment": conditional_alignment,
        },
        diagnostics=dict(diagnostics),
        capabilities={
            "predicted_noise",
            "target_noise",
            "reference_projection",
            "conditional_alignment",
        },
    )


def build_trajectory_artifacts(
    *,
    sample_id: str | None,
    category: str | None,
    representation: dict[str, object],
    reconstruction: torch.Tensor,
    trajectory: list[torch.Tensor],
    step_updates: list[torch.Tensor],
    step_costs: list[torch.Tensor],
    diagnostics: dict[str, Any],
) -> NormalityArtifacts:
    return NormalityArtifacts(
        context={
            "sample_id": sample_id,
            "category": category,
            "mode": "inference",
        },
        representation=representation,
        primary={"reconstruction": reconstruction},
        auxiliary={
            "trajectory": trajectory,
            "step_updates": step_updates,
            "step_costs": step_costs,
        },
        diagnostics=dict(diagnostics),
        capabilities={"trajectory", "step_costs"},
    )


def build_latent_reconstruction_artifacts(
    *,
    sample_id: str | None,
    category: str | None,
    representation: dict[str, object],
    latent_reconstruction: torch.Tensor,
    latent_target: torch.Tensor,
    reconstruction: torch.Tensor | None,
    diagnostics: dict[str, Any],
) -> NormalityArtifacts:
    primary = {"latent_reconstruction": latent_reconstruction}
    if reconstruction is not None:
        primary["reconstruction"] = reconstruction
    return NormalityArtifacts(
        context={
            "sample_id": sample_id,
            "category": category,
            "mode": "inference",
        },
        representation=representation,
        primary=primary,
        auxiliary={"latent_target": latent_target},
        diagnostics=dict(diagnostics),
        capabilities=set(),
    )


def build_conditioned_score_artifacts(
    *,
    sample_id: str | None,
    category: str | None,
    representation: dict[str, object],
    score_map: torch.Tensor,
    conditioning: dict[str, object],
    diagnostics: dict[str, Any],
) -> NormalityArtifacts:
    return NormalityArtifacts(
        context={
            "sample_id": sample_id,
            "category": category,
            "mode": "inference",
            "conditioning": dict(conditioning),
        },
        representation=representation,
        auxiliary={"score_map": score_map},
        diagnostics=dict(diagnostics),
        capabilities=set(),
    )


def build_reconstruction_free_score_artifacts(
    *,
    sample_id: str | None,
    category: str | None,
    representation: dict[str, object],
    score_map: torch.Tensor,
    score_features: torch.Tensor,
    diagnostics: dict[str, Any],
) -> NormalityArtifacts:
    return NormalityArtifacts(
        context={
            "sample_id": sample_id,
            "category": category,
            "mode": "inference",
        },
        representation=representation,
        auxiliary={
            "score_map": score_map,
            "score_features": score_features,
        },
        diagnostics=dict(diagnostics),
        capabilities=set(),
    )
