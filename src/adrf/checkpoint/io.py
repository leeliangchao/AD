"""Minimal checkpoint utilities based on model state dicts."""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
from torch import nn


def save_model_checkpoint(model: object, path: str | Path) -> bool:
    """Save a model state dict when the model has trainable parameters."""

    if not isinstance(model, nn.Module):
        warnings.warn("Model does not expose a state_dict; skipping checkpoint save.", stacklevel=2)
        return False

    if not any(parameter.requires_grad for parameter in model.parameters()):
        warnings.warn("Model has no trainable parameters; skipping checkpoint save.", stacklevel=2)
        return False

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_normalize_state_dict_keys(model.state_dict()), checkpoint_path)
    return True


def load_model_checkpoint(model: object, path: str | Path) -> bool:
    """Load a checkpoint into a compatible trainable model."""

    if not isinstance(model, nn.Module):
        warnings.warn("Model does not expose a state_dict; skipping checkpoint load.", stacklevel=2)
        return False

    checkpoint_path = Path(path)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(_normalize_state_dict_keys(state_dict))
    return True


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip DDP-style `.module.` path segments from serialized state dict keys."""

    return {
        key.replace(".module.", "."): value
        for key, value in state_dict.items()
    }
