"""Tests for internal diffusion task helpers."""

from __future__ import annotations

import torch

from adrf.normality.diffusion_tasks import (
    build_conditional_reconstruction_artifacts,
    build_reconstruction_artifacts,
    build_trajectory_artifacts,
)


def test_build_reconstruction_artifacts_exposes_canonical_payload_with_legacy_noise_capabilities() -> None:
    artifacts = build_reconstruction_artifacts(
        sample_id="sample-001",
        category="bottle",
        representation={"space": "pixel"},
        reconstruction=torch.zeros(3, 4, 4),
        predicted_noise=torch.ones(3, 4, 4),
        target_noise=torch.full((3, 4, 4), 2.0),
        diagnostics={"num_train_timesteps": 32},
    )

    assert artifacts.get_primary("reconstruction").shape == (3, 4, 4)
    assert artifacts.get_aux("predicted_noise").shape == (3, 4, 4)
    assert artifacts.get_aux("target_noise").shape == (3, 4, 4)
    assert artifacts.capabilities == {"predicted_noise", "target_noise"}


def test_build_conditional_reconstruction_artifacts_keeps_reference_contract() -> None:
    artifacts = build_conditional_reconstruction_artifacts(
        sample_id="sample-001",
        category="bottle",
        representation={"space": "pixel"},
        reconstruction=torch.zeros(3, 4, 4),
        reference_projection=torch.ones(3, 4, 4),
        predicted_noise=torch.zeros(3, 4, 4),
        target_noise=torch.zeros(3, 4, 4),
        conditional_alignment=torch.ones(4, 4),
        diagnostics={"num_train_timesteps": 32},
    )

    assert artifacts.get_primary("reconstruction").shape == (3, 4, 4)
    assert artifacts.get_primary("reference_projection").shape == (3, 4, 4)
    assert artifacts.get_aux("conditional_alignment").shape == (4, 4)
    assert artifacts.capabilities == {
        "predicted_noise",
        "target_noise",
        "reference_projection",
        "conditional_alignment",
    }


def test_build_trajectory_artifacts_exposes_canonical_payload_with_legacy_process_capabilities() -> None:
    artifacts = build_trajectory_artifacts(
        sample_id="sample-001",
        category="bottle",
        representation={"space": "pixel"},
        reconstruction=torch.zeros(3, 4, 4),
        trajectory=[torch.zeros(3, 4, 4), torch.ones(3, 4, 4)],
        step_updates=[torch.ones(3, 4, 4), torch.zeros(3, 4, 4)],
        step_costs=[torch.ones(4, 4), torch.zeros(4, 4)],
        diagnostics={"num_steps": 2},
    )

    assert artifacts.get_primary("reconstruction").shape == (3, 4, 4)
    assert len(artifacts.get_aux("trajectory")) == 2
    assert len(artifacts.get_aux("step_updates")) == 2
    assert len(artifacts.get_aux("step_costs")) == 2
    assert artifacts.capabilities == {"trajectory", "step_costs"}
