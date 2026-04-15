from __future__ import annotations

from types import MethodType

from torch import nn

from adrf.normality.autoencoder import AutoEncoderNormality
from adrf.normality.diffusion_basic import DiffusionBasicNormality
from adrf.normality.diffusion_inversion_basic import DiffusionInversionBasicNormality
from adrf.normality.reference_basic import ReferenceBasicNormality
from adrf.normality.reference_diffusion_basic import ReferenceDiffusionBasicNormality
from adrf.normality.runtime import resolve_normality_runtime_spec
from adrf.utils.distributed import DistributedRuntimeContext
from adrf.utils.runtime import configure_trainable_runtime


class _Impostor:
    def __init__(self) -> None:
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()
        self._forward_impl = lambda inputs: (inputs, inputs)
        self.fit = MethodType(lambda self, representations, samples=None: None, self)


def test_resolve_normality_runtime_spec_maps_known_models_to_explicit_specs() -> None:
    assert resolve_normality_runtime_spec(AutoEncoderNormality()).fit_wrapper_id == "autoencoder"
    assert resolve_normality_runtime_spec(DiffusionBasicNormality()).fit_wrapper_id == "diffusion"
    assert resolve_normality_runtime_spec(DiffusionInversionBasicNormality()).fit_wrapper_id == "diffusion"
    assert resolve_normality_runtime_spec(ReferenceBasicNormality()).fit_wrapper_id == "reference_basic"
    assert resolve_normality_runtime_spec(ReferenceDiffusionBasicNormality()).fit_wrapper_id == "reference_diffusion"


def test_resolve_normality_runtime_spec_exposes_expected_distributed_modules() -> None:
    assert resolve_normality_runtime_spec(AutoEncoderNormality()).distributed_module_names == ("encoder", "decoder")
    assert resolve_normality_runtime_spec(DiffusionBasicNormality()).distributed_module_names == ("denoiser",)
    assert resolve_normality_runtime_spec(ReferenceBasicNormality()).distributed_module_names == ("conditional_model",)
    assert (
        resolve_normality_runtime_spec(ReferenceDiffusionBasicNormality()).distributed_module_names
        == ("conditional_denoiser",)
    )


def test_resolve_normality_runtime_spec_rejects_duck_typed_impostor() -> None:
    assert resolve_normality_runtime_spec(_Impostor()) is None


def test_configure_trainable_runtime_does_not_patch_duck_typed_impostor() -> None:
    impostor = _Impostor()
    original_fit = impostor.fit

    configure_trainable_runtime(
        impostor,
        device=nn.Parameter().device,
        amp_enabled=False,
        distributed_context=DistributedRuntimeContext(),
    )

    assert impostor.fit is original_fit
