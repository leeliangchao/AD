"""Normality models for the MVP anomaly detection framework."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "AutoEncoderNormality": ("adrf.normality.autoencoder", "AutoEncoderNormality"),
    "BaseNormalityModel": ("adrf.normality.base", "BaseNormalityModel"),
    "DiffusionBasicNormality": ("adrf.normality.diffusion_basic", "DiffusionBasicNormality"),
    "DiffusionInversionBasicNormality": ("adrf.normality.diffusion_inversion_basic", "DiffusionInversionBasicNormality"),
    "FeatureMemoryNormality": ("adrf.normality.feature_memory", "FeatureMemoryNormality"),
    "ReferenceBasicNormality": ("adrf.normality.reference_basic", "ReferenceBasicNormality"),
    "ReferenceDiffusionBasicNormality": ("adrf.normality.reference_diffusion_basic", "ReferenceDiffusionBasicNormality"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> object:
    """Lazily resolve normality exports to avoid package-level import cycles."""

    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'adrf.normality' has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy exports for interactive discovery."""

    return sorted(list(globals().keys()) + __all__)
