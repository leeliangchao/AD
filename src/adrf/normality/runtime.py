from __future__ import annotations

from dataclasses import dataclass

from adrf.normality.autoencoder import AutoEncoderNormality
from adrf.normality.diffusion_basic import DiffusionBasicNormality
from adrf.normality.reference_basic import ReferenceBasicNormality
from adrf.normality.reference_diffusion_basic import ReferenceDiffusionBasicNormality


@dataclass(frozen=True, slots=True)
class NormalityRuntimeSpec:
    fit_wrapper_id: str
    distributed_module_names: tuple[str, ...]


def resolve_normality_runtime_spec(model: object) -> NormalityRuntimeSpec | None:
    if isinstance(model, AutoEncoderNormality):
        return NormalityRuntimeSpec(
            fit_wrapper_id="autoencoder",
            distributed_module_names=("encoder", "decoder"),
        )
    if isinstance(model, ReferenceDiffusionBasicNormality):
        return NormalityRuntimeSpec(
            fit_wrapper_id="reference_diffusion",
            distributed_module_names=("conditional_denoiser",),
        )
    if isinstance(model, ReferenceBasicNormality):
        return NormalityRuntimeSpec(
            fit_wrapper_id="reference_basic",
            distributed_module_names=("conditional_model",),
        )
    if isinstance(model, DiffusionBasicNormality):
        return NormalityRuntimeSpec(
            fit_wrapper_id="diffusion",
            distributed_module_names=("denoiser",),
        )
    return None
