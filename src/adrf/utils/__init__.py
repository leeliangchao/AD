"""Utility helpers for configuration and other framework support code."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "all_gather_objects": ("adrf.utils.distributed", "all_gather_objects"),
    "instantiate_component": ("adrf.utils.config", "instantiate_component"),
    "load_runtime_profile": ("adrf.utils.runtime", "load_runtime_profile"),
    "load_yaml_config": ("adrf.utils.config", "load_yaml_config"),
    "resolve_distributed_context": ("adrf.utils.distributed", "resolve_distributed_context"),
    "resolve_dataloader_runtime": ("adrf.utils.runtime", "resolve_dataloader_runtime"),
    "resolve_device": ("adrf.utils.device", "resolve_device"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> object:
    """Lazily resolve utility exports to avoid importing optional runtime dependencies eagerly."""

    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'adrf.utils' has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the exported lazy attributes for interactive discovery."""

    return sorted(list(globals().keys()) + __all__)
