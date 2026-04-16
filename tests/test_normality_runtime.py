from __future__ import annotations

from types import MethodType

import torch
from torch import nn

from adrf.core.sample import Sample
from adrf.normality.autoencoder import AutoEncoderNormality
from adrf.normality.diffusion_basic import DiffusionBasicNormality
from adrf.normality.diffusion_inversion_basic import DiffusionInversionBasicNormality
from adrf.normality.reference_basic import ReferenceBasicNormality
from adrf.normality.reference_diffusion_basic import ReferenceDiffusionBasicNormality
from adrf.normality.runtime import resolve_normality_runtime_spec
from adrf.normality.state import NormalityRuntimeState
from adrf.utils.distributed import DistributedRuntimeContext
from adrf.utils.runtime import _wrap_fit_for_runtime, configure_trainable_runtime

from support.representation_builders import make_pixel_output


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


def test_diffusion_basic_initializes_explicit_runtime_state_with_legacy_aliases() -> None:
    model = DiffusionBasicNormality()

    assert isinstance(model.runtime, NormalityRuntimeState)
    assert model.runtime.device == model.runtime_device
    assert model.runtime.amp_enabled == model.amp_enabled
    assert model.runtime.grad_scaler is model.grad_scaler


def test_autoencoder_initializes_explicit_runtime_state_with_legacy_aliases() -> None:
    model = AutoEncoderNormality()

    assert isinstance(model.runtime, NormalityRuntimeState)
    assert model.runtime.device == model.runtime_device
    assert model.runtime.amp_enabled == model.amp_enabled
    assert model.runtime.grad_scaler is model.grad_scaler


def test_reference_basic_initializes_explicit_runtime_state_with_legacy_aliases() -> None:
    model = ReferenceBasicNormality()

    assert isinstance(model.runtime, NormalityRuntimeState)
    assert model.runtime.device == model.runtime_device
    assert model.runtime.amp_enabled == model.amp_enabled
    assert model.runtime.grad_scaler is model.grad_scaler


def test_reference_diffusion_initializes_explicit_runtime_state_with_legacy_aliases() -> None:
    model = ReferenceDiffusionBasicNormality()

    assert isinstance(model.runtime, NormalityRuntimeState)
    assert model.runtime.device == model.runtime_device
    assert model.runtime.amp_enabled == model.amp_enabled
    assert model.runtime.grad_scaler is model.grad_scaler


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


def test_configure_trainable_runtime_attaches_explicit_runtime_state() -> None:
    model = AutoEncoderNormality(input_channels=3, hidden_channels=4, latent_channels=8)
    context = DistributedRuntimeContext()

    configure_trainable_runtime(
        model,
        device=nn.Parameter().device,
        amp_enabled=False,
        distributed_context=context,
    )

    assert isinstance(model.runtime, NormalityRuntimeState)
    assert model.runtime.device.type == "cpu"
    assert model.runtime.amp_enabled is False
    assert model.runtime.distributed_context is context
    assert model.runtime.distributed_training_enabled is False
    assert model.runtime_device == model.runtime.device
    assert model.amp_enabled == model.runtime.amp_enabled
    assert model.distributed_context is model.runtime.distributed_context


def test_configure_trainable_runtime_preserves_reference_diffusion_class_conditioning_fit() -> None:
    generator = torch.Generator().manual_seed(0)
    samples = [
        Sample(
            image=torch.rand(3, 16, 16, generator=generator),
            reference=torch.rand(3, 16, 16, generator=generator),
            sample_id="sample-001",
            category="bottle",
        ),
        Sample(
            image=torch.rand(3, 16, 16, generator=generator),
            reference=torch.rand(3, 16, 16, generator=generator),
            sample_id="sample-002",
            category="capsule",
        ),
    ]
    representations = [make_pixel_output(sample.image, sample_id=sample.sample_id or "sample") for sample in samples]
    model = ReferenceDiffusionBasicNormality(
        input_channels=3,
        hidden_channels=8,
        time_embed_dim=32,
        num_train_timesteps=32,
        learning_rate=1e-3,
        epochs=1,
        batch_size=2,
        noise_level=0.2,
        num_classes=2,
    )

    configure_trainable_runtime(
        model,
        device=torch.device("cpu"),
        amp_enabled=False,
        distributed_context=DistributedRuntimeContext(),
    )
    model.fit(representations, samples)

    assert model.class_to_index == {"bottle": 0, "capsule": 1}


def test_runtime_fit_wrapper_preserves_reference_diffusion_class_conditioning_fit() -> None:
    generator = torch.Generator().manual_seed(0)
    samples = [
        Sample(
            image=torch.rand(3, 16, 16, generator=generator),
            reference=torch.rand(3, 16, 16, generator=generator),
            sample_id="sample-001",
            category="bottle",
        ),
        Sample(
            image=torch.rand(3, 16, 16, generator=generator),
            reference=torch.rand(3, 16, 16, generator=generator),
            sample_id="sample-002",
            category="capsule",
        ),
    ]
    representations = [make_pixel_output(sample.image, sample_id=sample.sample_id or "sample") for sample in samples]
    model = ReferenceDiffusionBasicNormality(
        input_channels=3,
        hidden_channels=8,
        time_embed_dim=32,
        num_train_timesteps=32,
        learning_rate=1e-3,
        epochs=1,
        batch_size=2,
        noise_level=0.2,
        num_classes=2,
    )

    _wrap_fit_for_runtime(model)
    model.fit(representations, samples)

    assert model.class_to_index == {"bottle": 0, "capsule": 1}


def test_runtime_fit_wrapper_preserves_reference_diffusion_diffusers_class_conditioning_fit() -> None:
    generator = torch.Generator().manual_seed(0)
    samples = [
        Sample(
            image=torch.rand(3, 16, 16, generator=generator),
            reference=torch.rand(3, 16, 16, generator=generator),
            sample_id="sample-001",
            category="bottle",
        ),
        Sample(
            image=torch.rand(3, 16, 16, generator=generator),
            reference=torch.rand(3, 16, 16, generator=generator),
            sample_id="sample-002",
            category="capsule",
        ),
    ]
    representations = [make_pixel_output(sample.image, sample_id=sample.sample_id or "sample") for sample in samples]
    model = ReferenceDiffusionBasicNormality(
        input_channels=3,
        hidden_channels=8,
        time_embed_dim=32,
        num_train_timesteps=32,
        learning_rate=1e-3,
        epochs=1,
        batch_size=2,
        noise_level=0.2,
        num_classes=2,
        backend="diffusers",
    )

    _wrap_fit_for_runtime(model)
    model.fit(representations, samples)

    assert model.class_to_index == {"bottle": 0, "capsule": 1}
