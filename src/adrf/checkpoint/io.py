"""Minimal checkpoint utilities based on model state dicts."""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
from torch import nn


def save_model_checkpoint(model: object, path: str | Path) -> bool:
    """Save a model state dict when the model has trainable parameters."""

    resolved_model = _resolve_checkpoint_model(model)
    if resolved_model is None:
        warnings.warn("Model does not expose a state_dict; skipping checkpoint save.", stacklevel=2)
        return False

    if not any(parameter.requires_grad for parameter in resolved_model.parameters()):
        warnings.warn("Model has no trainable parameters; skipping checkpoint save.", stacklevel=2)
        return False

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(resolved_model.state_dict(), checkpoint_path)
    return True


def load_model_checkpoint(model: object, path: str | Path) -> bool:
    """Load a checkpoint into a compatible trainable model."""

    resolved_model = _resolve_checkpoint_model(model)
    if resolved_model is None:
        warnings.warn("Model does not expose a state_dict; skipping checkpoint load.", stacklevel=2)
        return False

    checkpoint_path = Path(path)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint at '{checkpoint_path}' must contain a state dict mapping.")
    normalized = _normalize_state_dict_keys(state_dict, expected_keys=set(resolved_model.state_dict()))
    resolved_model.load_state_dict(normalized)
    return True


def _normalize_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
    *,
    expected_keys: set[str],
) -> dict[str, torch.Tensor]:
    """Strip DDP wrapper `module` segments only when they map onto expected model keys."""

    return {
        _resolve_normalized_key(key, expected_keys=expected_keys): value
        for key, value in state_dict.items()
    }


def _resolve_normalized_key(key: str, *, expected_keys: set[str]) -> str:
    """Map a serialized parameter key onto one expected by the destination model."""

    if key in expected_keys:
        return key

    variants = _module_stripped_variants(key)
    matches = [candidate for candidate in variants if candidate in expected_keys]
    if len(matches) == 1:
        return matches[0]
    return key


def _module_stripped_variants(key: str) -> set[str]:
    """Generate candidate keys by removing one or more DDP-style `module` path segments."""

    segments = key.split(".")
    variants = {key}
    frontier = [segments]
    seen = {tuple(segments)}

    while frontier:
        current = frontier.pop()
        for index, segment in enumerate(current):
            if segment != "module":
                continue
            candidate = current[:index] + current[index + 1 :]
            if not candidate:
                continue
            candidate_tuple = tuple(candidate)
            if candidate_tuple in seen:
                continue
            seen.add(candidate_tuple)
            candidate_key = ".".join(candidate)
            variants.add(candidate_key)
            frontier.append(candidate)

    return variants


def _resolve_checkpoint_model(model: object) -> nn.Module | None:
    """Return the trainable module that should own checkpoint serialization."""

    if isinstance(model, nn.Module):
        return model

    wrapped_model = getattr(model, "representation", None)
    if isinstance(wrapped_model, nn.Module):
        return wrapped_model

    return None
