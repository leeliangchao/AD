"""Tests for the stronger official baseline filling contract."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

from adrf.ablation.matrix import AblationMatrix
from adrf.core.sample import Sample
from adrf.normality.autoencoder import AutoEncoderNormality
from adrf.normality.diffusion_basic import DiffusionBasicNormality
from adrf.normality.diffusion_inversion_basic import DiffusionInversionBasicNormality
from adrf.normality.reference_basic import ReferenceBasicNormality
from adrf.normality.reference_diffusion_basic import ReferenceDiffusionBasicNormality
from adrf.utils.config import load_yaml_config

sys.path.insert(0, str(Path(__file__).parent))

from support.representation_builders import make_pixel_output


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILLED_SMOKE_MATRIX = PROJECT_ROOT / "configs" / "ablation" / "paper_baseline_matrix_official_v1_filled_smoke.yaml"
FILLED_FULL_MATRIX = PROJECT_ROOT / "configs" / "ablation" / "paper_baseline_matrix_official_v1_filled.yaml"


def test_filled_experiment_configs_expose_explicit_capacity_parameters() -> None:
    """Filled experiment configs should stop relying on minimal hard-coded model capacity."""

    config_expectations = {
        "configs/experiment/recon_baseline.yaml": {"base_channels", "channel_mults", "num_blocks_per_stage"},
        "configs/experiment/diffusion_baseline.yaml": {"base_channels", "channel_mults", "num_res_blocks"},
        "configs/experiment/diffusion_pathcost_baseline.yaml": {
            "base_channels",
            "channel_mults",
            "num_res_blocks",
            "initial_noise_scale",
        },
        "configs/experiment/diffusion_direction_baseline.yaml": {
            "base_channels",
            "channel_mults",
            "num_res_blocks",
            "initial_noise_scale",
        },
        "configs/experiment/reference_baseline.yaml": {
            "base_channels",
            "channel_mults",
            "num_res_blocks",
            "condition_channels",
        },
        "configs/experiment/reference_diffusion_baseline.yaml": {
            "base_channels",
            "channel_mults",
            "num_res_blocks",
            "condition_channels",
            "time_embed_dim",
            "num_train_timesteps",
        },
    }
    config_expectations["configs/experiment/diffusion_baseline.yaml"] |= {"time_embed_dim", "num_train_timesteps"}
    config_expectations["configs/experiment/diffusion_pathcost_baseline.yaml"] |= {
        "time_embed_dim",
        "num_train_timesteps",
    }
    config_expectations["configs/experiment/diffusion_direction_baseline.yaml"] |= {
        "time_embed_dim",
        "num_train_timesteps",
    }

    for relative_path, expected_keys in config_expectations.items():
        config = load_yaml_config(PROJECT_ROOT / relative_path)
        normality_params = config["normality"]["params"]
        assert expected_keys <= set(normality_params)


def test_filled_diffusion_family_accepts_capacity_and_process_parameters() -> None:
    """Diffusion families should accept explicit structure/process knobs without changing artifact keys."""

    torch.manual_seed(0)
    representations = [
        make_pixel_output(torch.rand(3, 16, 16), sample_id=f"diffusion-{index:03d}")
        for index in range(2)
    ]
    diffusion_basic = DiffusionBasicNormality(
        input_channels=3,
        base_channels=12,
        channel_mults=[1, 2],
        num_res_blocks=2,
        time_embed_dim=64,
        num_train_timesteps=64,
        learning_rate=1e-3,
        epochs=1,
        batch_size=2,
        noise_level=0.2,
    )
    diffusion_basic.fit(representations)
    diffusion_artifacts = diffusion_basic.infer(
        Sample(image=representations[0].tensor, sample_id="diffusion"),
        representations[0],
    )

    assert diffusion_basic.base_channels == 12
    assert diffusion_basic.channel_mults == (1, 2)
    assert diffusion_basic.num_res_blocks == 2
    assert diffusion_basic.time_embed_dim == 64
    assert diffusion_basic.num_train_timesteps == 64
    assert diffusion_artifacts.capabilities == {"predicted_noise", "target_noise"}
    assert diffusion_artifacts.get_aux("predicted_noise").shape == (3, 16, 16)
    assert diffusion_artifacts.get_aux("target_noise").shape == (3, 16, 16)

    diffusion_inversion = DiffusionInversionBasicNormality(
        input_channels=3,
        base_channels=12,
        channel_mults=[1, 2],
        num_res_blocks=2,
        time_embed_dim=64,
        num_train_timesteps=64,
        learning_rate=1e-3,
        epochs=1,
        batch_size=2,
        noise_level=0.2,
        num_steps=5,
        step_size=0.1,
        initial_noise_scale=0.15,
    )
    diffusion_inversion.fit(representations)
    inversion_artifacts = diffusion_inversion.infer(
        Sample(image=representations[0].tensor, sample_id="process"),
        representations[0],
    )

    assert diffusion_inversion.initial_noise_scale == 0.15
    assert diffusion_inversion.time_embed_dim == 64
    assert diffusion_inversion.num_train_timesteps == 64
    assert inversion_artifacts.capabilities == {"trajectory", "step_costs"}
    assert len(inversion_artifacts.get_aux("trajectory")) == 5
    assert len(inversion_artifacts.get_aux("step_costs")) == 5

    reference_samples = [
        Sample(image=torch.rand(3, 16, 16), reference=torch.rand(3, 16, 16), sample_id="ref-001"),
        Sample(image=torch.rand(3, 16, 16), reference=torch.rand(3, 16, 16), sample_id="ref-002"),
    ]
    reference_representations = [
        make_pixel_output(sample.image, sample_id=sample.sample_id or "sample") for sample in reference_samples
    ]
    reference_diffusion = ReferenceDiffusionBasicNormality(
        input_channels=3,
        base_channels=12,
        condition_channels=10,
        channel_mults=[1, 2],
        num_res_blocks=2,
        time_embed_dim=64,
        num_train_timesteps=64,
        learning_rate=1e-3,
        epochs=1,
        batch_size=2,
        noise_level=0.2,
    )
    reference_diffusion.fit(reference_representations, reference_samples)
    reference_artifacts = reference_diffusion.infer(reference_samples[0], reference_representations[0])

    assert reference_diffusion.condition_channels == 10
    assert reference_diffusion.time_embed_dim == 64
    assert reference_diffusion.num_train_timesteps == 64
    assert reference_artifacts.capabilities == {
        "predicted_noise",
        "target_noise",
        "reference_projection",
        "conditional_alignment",
    }
    assert reference_artifacts.get_aux("predicted_noise").shape == (3, 16, 16)
    assert reference_artifacts.get_primary("reference_projection").shape == (3, 16, 16)
    assert reference_artifacts.representation["space"] == "pixel"
    assert reference_artifacts.representation["sample_id"] == "ref-001"


def test_filled_classical_and_conditional_baselines_preserve_artifact_contracts() -> None:
    """Autoencoder and conditional classical baselines should expose stronger capacity without artifact drift."""

    torch.manual_seed(0)
    representations = [
        make_pixel_output(torch.rand(3, 32, 32), sample_id=f"autoencoder-{index:03d}")
        for index in range(2)
    ]
    autoencoder = AutoEncoderNormality(
        input_channels=3,
        base_channels=8,
        channel_mults=[1, 2],
        latent_channels=24,
        num_blocks_per_stage=2,
        learning_rate=1e-3,
        epochs=1,
        batch_size=2,
    )
    autoencoder.fit(representations)
    autoencoder_artifacts = autoencoder.infer(
        Sample(image=representations[0].tensor, sample_id="ae"),
        representations[0],
    )

    assert autoencoder.base_channels == 8
    assert autoencoder.channel_mults == (1, 2)
    assert autoencoder.num_blocks_per_stage == 2
    assert autoencoder_artifacts.capabilities == {"projection", "reconstruction"}
    assert autoencoder_artifacts.get_primary("reconstruction").shape == (3, 32, 32)

    reference_samples = [
        Sample(image=torch.rand(3, 16, 16), reference=torch.rand(3, 16, 16), sample_id="rb-001"),
        Sample(image=torch.rand(3, 16, 16), reference=torch.rand(3, 16, 16), sample_id="rb-002"),
    ]
    reference_representations = [
        make_pixel_output(sample.image, sample_id=sample.sample_id or "sample") for sample in reference_samples
    ]
    reference_basic = ReferenceBasicNormality(
        input_channels=3,
        base_channels=10,
        condition_channels=8,
        channel_mults=[1, 2],
        num_res_blocks=2,
        learning_rate=1e-3,
        epochs=1,
        batch_size=2,
    )
    reference_basic.fit(reference_representations, reference_samples)
    reference_artifacts = reference_basic.infer(reference_samples[0], reference_representations[0])

    assert reference_basic.base_channels == 10
    assert reference_basic.condition_channels == 8
    assert reference_basic.channel_mults == (1, 2)
    assert reference_artifacts.capabilities == {"reference_projection", "conditional_alignment"}
    assert reference_artifacts.get_primary("reference_projection").shape == (3, 16, 16)
    assert reference_artifacts.get_aux("conditional_alignment").shape == (16, 16)
    assert reference_artifacts.representation["space"] == "pixel"
    assert reference_artifacts.representation["sample_id"] == "rb-001"


def test_filled_matrix_configs_expand_to_expected_contract_sizes() -> None:
    """Filled matrix configs should preserve the official comparison shape while adding stronger params."""

    smoke_specs = AblationMatrix.from_yaml(FILLED_SMOKE_MATRIX).expand()
    full_specs = AblationMatrix.from_yaml(FILLED_FULL_MATRIX).expand()

    assert len(smoke_specs) == 7
    assert {spec["dataset"] for spec in smoke_specs} == {"mvtec_bottle"}
    assert {spec["seed"] for spec in smoke_specs} == {0}

    assert len(full_specs) == 63
    assert {spec["dataset"] for spec in full_specs} == {"mvtec_bottle", "mvtec_capsule", "mvtec_grid"}
    assert {spec["seed"] for spec in full_specs} == {0, 1, 2}

    diffusion_smoke_specs = [
        spec for spec in smoke_specs if spec["normality"] in {"diffusion_basic", "diffusion_inversion_basic", "reference_diffusion_basic"}
    ]
    assert diffusion_smoke_specs
    for spec in diffusion_smoke_specs:
        params = spec["config"]["normality"]["params"]
        assert params["time_embed_dim"] == 64
        assert params["num_train_timesteps"] == 100
