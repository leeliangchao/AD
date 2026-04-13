"""Minimal YAML configuration helpers for component construction."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from adrf.registry.registry import Registry


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""

    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML config at '{config_path}' must load to a mapping.")
    return data


def instantiate_component(
    spec: Mapping[str, Any],
    registry: Registry,
    group: str | None = None,
    **overrides: Any,
) -> Any:
    """Instantiate a registry entry from a minimal config specification."""

    if not isinstance(spec, Mapping):
        raise TypeError("Component spec must be a mapping.")

    resolved_group = group or spec.get("group")
    if not resolved_group:
        raise ValueError("Component spec must define a group or receive one explicitly.")

    name = spec.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Component spec must define a non-empty 'name'.")

    raw_params = spec.get("params", {})
    if not isinstance(raw_params, Mapping):
        raise TypeError("Component spec 'params' must be a mapping when provided.")

    params = dict(raw_params)
    params.update(overrides)

    target = registry.get(resolved_group, name)
    if callable(target):
        return target(**params)
    if params:
        raise TypeError(
            f"Registered object '{name}' in group '{resolved_group}' is not callable and "
            "cannot be instantiated with parameters."
        )
    return target
