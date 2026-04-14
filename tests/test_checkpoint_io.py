"""Tests for minimal checkpoint save/load helpers."""

from pathlib import Path

import pytest
import torch
from torch import nn

from adrf.checkpoint.io import load_model_checkpoint, save_model_checkpoint
from adrf.representation.base import BaseRepresentation
from adrf.representation.contracts import RepresentationProvenance
from adrf.normality.autoencoder import AutoEncoderNormality
from adrf.normality.feature_memory import FeatureMemoryNormality


class _TrainableToyRepresentation(BaseRepresentation):
    space = "feature"
    trainable = True

    def __init__(self) -> None:
        super().__init__(input_image_size=(32, 32), input_normalize=False)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def _encode_tensor_batch(self, batch: torch.Tensor) -> torch.Tensor:
        return batch[:, :1, :, :] * self.scale

    def describe(self) -> RepresentationProvenance:
        return RepresentationProvenance(
            representation_name="feature",
            backbone_name="toy",
            weights_source=None,
            feature_layer="scale",
            pooling=None,
            trainable=True,
            frozen_submodules=(),
            input_image_size=self.input_image_size,
            input_normalize=self.input_normalize,
            normalize_mean=None,
            normalize_std=None,
            code_version="tests",
            config_fingerprint="checkpoint-io-toy-feature",
        )


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


def test_checkpoint_io_saves_and_loads_trainable_representation_module(tmp_path: Path) -> None:
    """Trainable representation modules should also round-trip through checkpoints."""

    model = _TrainableToyRepresentation()
    checkpoint_path = tmp_path / "representation.pt"

    saved = save_model_checkpoint(model, checkpoint_path)
    restored = _TrainableToyRepresentation()
    restored.scale.data.zero_()
    loaded = load_model_checkpoint(restored, checkpoint_path)

    assert saved is True
    assert loaded is True
    assert checkpoint_path.exists()
    assert torch.allclose(model.scale.detach(), restored.scale.detach())


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
