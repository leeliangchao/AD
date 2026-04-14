"""Tests for minimal checkpoint save/load helpers."""

from pathlib import Path

import pytest
import torch

from adrf.checkpoint.io import load_model_checkpoint, save_model_checkpoint
from adrf.normality.autoencoder import AutoEncoderNormality
from adrf.normality.feature_memory import FeatureMemoryNormality


def test_checkpoint_io_saves_and_loads_trainable_model(tmp_path: Path) -> None:
    """Trainable nn.Module normality models should round-trip through checkpoints."""

    model = AutoEncoderNormality(input_channels=3, hidden_channels=4, latent_channels=8)
    checkpoint_path = tmp_path / "normality.pt"

    saved = save_model_checkpoint(model, checkpoint_path)
    restored = AutoEncoderNormality(input_channels=3, hidden_channels=4, latent_channels=8)
    loaded = load_model_checkpoint(restored, checkpoint_path)

    assert saved is True
    assert loaded is True
    assert checkpoint_path.exists()
    first_saved = next(model.parameters()).detach()
    first_restored = next(restored.parameters()).detach()
    assert torch.allclose(first_saved, first_restored)


def test_checkpoint_io_skips_non_trainable_models_with_warning(tmp_path: Path) -> None:
    """Models without state_dict support should be skipped with a warning."""

    checkpoint_path = tmp_path / "feature_memory.pt"

    with pytest.warns(UserWarning, match="state_dict"):
        saved = save_model_checkpoint(FeatureMemoryNormality(), checkpoint_path)

    assert saved is False
    assert checkpoint_path.exists() is False


def test_checkpoint_io_loads_state_dicts_saved_from_ddp_wrapped_submodules(tmp_path: Path) -> None:
    """Checkpoint loading should tolerate `.module.` prefixes from DDP-wrapped submodules."""

    model = AutoEncoderNormality(input_channels=3, hidden_channels=4, latent_channels=8)
    checkpoint_path = tmp_path / "normality_ddp.pt"
    ddp_like_state = {
        key.replace("encoder.", "encoder.module.").replace("decoder.", "decoder.module."): value
        for key, value in model.state_dict().items()
    }
    torch.save(ddp_like_state, checkpoint_path)

    restored = AutoEncoderNormality(input_channels=3, hidden_channels=4, latent_channels=8)
    loaded = load_model_checkpoint(restored, checkpoint_path)

    assert loaded is True
    first_saved = next(model.parameters()).detach()
    first_restored = next(restored.parameters()).detach()
    assert torch.allclose(first_saved, first_restored)
