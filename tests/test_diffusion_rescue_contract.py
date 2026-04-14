"""Contract tests for the diffusion rescue baselines."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

from adrf.core.sample import Sample
from adrf.evidence.direction_mismatch import DirectionMismatchEvidence
from adrf.evidence.path_cost import PathCostEvidence
from adrf.normality.diffusion_basic import DiffusionBasicNormality
from adrf.normality.diffusion_inversion_basic import DiffusionInversionBasicNormality
from adrf.utils.config import load_yaml_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

from support.representation_builders import make_pixel_output


def test_diffusion_rescue_configs_expose_explicit_conditioning_and_process_params() -> None:
    """Rescue configs should surface the new diffusion knobs explicitly."""

    diffusion_params = load_yaml_config(PROJECT_ROOT / "configs" / "experiment" / "diffusion_baseline.yaml")["normality"]["params"]
    pathcost_params = load_yaml_config(PROJECT_ROOT / "configs" / "experiment" / "diffusion_pathcost_baseline.yaml")["normality"]["params"]
    direction_params = load_yaml_config(PROJECT_ROOT / "configs" / "experiment" / "diffusion_direction_baseline.yaml")["normality"]["params"]
    smoke_normality = load_yaml_config(PROJECT_ROOT / "configs" / "ablation" / "paper_baseline_matrix_official_v1_filled_smoke.yaml")[
        "normality"
    ]

    assert {"base_channels", "channel_mults", "num_res_blocks", "time_embed_dim", "conditioning_hidden_dim"} <= set(
        diffusion_params
    )
    assert {
        "base_channels",
        "channel_mults",
        "num_res_blocks",
        "time_embed_dim",
        "conditioning_hidden_dim",
        "num_steps",
        "step_size",
        "initial_noise_scale",
        "rollout_gain",
        "denoised_blend",
    } <= set(pathcost_params)
    assert {
        "conditioning_hidden_dim",
        "rollout_gain",
        "denoised_blend",
    } <= set(direction_params)
    assert {"conditioning_hidden_dim"} <= set(smoke_normality["diffusion_basic"]["params"])
    assert {"conditioning_hidden_dim", "rollout_gain", "denoised_blend"} <= set(
        smoke_normality["diffusion_inversion_basic"]["params"]
    )


def test_diffusion_basic_rescue_preserves_noise_artifact_contract() -> None:
    """The rescued diffusion basic model should keep noise artifact semantics unchanged."""

    generator = torch.Generator().manual_seed(0)
    representations = [
        make_pixel_output(torch.rand(3, 16, 16, generator=generator), sample_id=f"train-{index:03d}")
        for index in range(2)
    ]
    model = DiffusionBasicNormality(
        input_channels=3,
        base_channels=12,
        channel_mults=[1, 2],
        num_res_blocks=2,
        time_embed_dim=64,
        conditioning_hidden_dim=96,
        num_train_timesteps=64,
        learning_rate=1e-3,
        epochs=1,
        batch_size=2,
        noise_level=0.2,
    )

    model.fit(representations)
    artifacts = model.infer(Sample(image=representations[0].tensor, sample_id="diff-basic"), representations[0])

    assert artifacts.capabilities == {"predicted_noise", "target_noise"}
    assert artifacts.get_aux("predicted_noise").shape == (3, 16, 16)
    assert artifacts.get_aux("target_noise").shape == (3, 16, 16)
    assert artifacts.get_diag("conditioning_hidden_dim") == 96
    assert artifacts.representation == representations[0].to_artifact_dict()


def test_diffusion_inversion_rescue_preserves_process_artifact_contract_and_evidence_reuse() -> None:
    """The rescued inversion model should keep process artifacts consumable by process evidence."""

    generator = torch.Generator().manual_seed(0)
    representations = [
        make_pixel_output(torch.rand(3, 16, 16, generator=generator), sample_id=f"train-{index:03d}")
        for index in range(2)
    ]
    sample = Sample(image=representations[0].tensor, sample_id="diff-process")
    model = DiffusionInversionBasicNormality(
        input_channels=3,
        base_channels=12,
        channel_mults=[1, 2],
        num_res_blocks=2,
        time_embed_dim=64,
        conditioning_hidden_dim=96,
        num_train_timesteps=64,
        learning_rate=1e-3,
        epochs=1,
        batch_size=2,
        noise_level=0.2,
        num_steps=5,
        step_size=0.15,
        initial_noise_scale=0.25,
        rollout_gain=1.1,
        denoised_blend=0.2,
    )

    model.fit(representations)
    artifacts = model.infer(sample, representations[0])

    trajectory = artifacts.get_aux("trajectory")
    step_costs = artifacts.get_aux("step_costs")
    assert artifacts.capabilities == {"trajectory", "step_costs"}
    assert len(trajectory) == len(step_costs) == 5
    assert trajectory[0].shape == trajectory[-1].shape == (3, 16, 16)
    assert step_costs[0].shape == step_costs[-1].shape == (16, 16)
    assert artifacts.get_diag("conditioning_hidden_dim") == 96
    assert artifacts.get_diag("rollout_gain") == 1.1
    assert artifacts.get_diag("denoised_blend") == 0.2
    assert artifacts.representation == representations[0].to_artifact_dict()

    path_prediction = PathCostEvidence(aggregator="mean").predict(sample, artifacts)
    direction_prediction = DirectionMismatchEvidence(aggregator="mean", direction_reduce="sum").predict(sample, artifacts)

    assert path_prediction["anomaly_map"].shape == (16, 16)
    assert isinstance(path_prediction["image_score"], float)
    assert direction_prediction["anomaly_map"].shape == (16, 16)
    assert isinstance(direction_prediction["image_score"], float)
