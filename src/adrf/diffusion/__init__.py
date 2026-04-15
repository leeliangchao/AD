"""Minimal diffusers adapter layer for AD diffusion backends."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "DiffusersNoisePredictorAdapter": ("adrf.diffusion.adapters", "DiffusersNoisePredictorAdapter"),
    "make_scheduler": ("adrf.diffusion.schedulers", "make_scheduler"),
    "make_unet_model": ("adrf.diffusion.models", "make_unet_model"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> object:
    """Lazily resolve diffusion exports to avoid package-level import cycles."""

    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'adrf.diffusion' has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy exports for interactive discovery."""

    return sorted(list(globals().keys()) + __all__)
