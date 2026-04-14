"""Runtime profile loading and lightweight device helpers."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from adrf.core.sample import Sample
from adrf.utils.config import load_yaml_config
from adrf.utils.distributed import DistributedRuntimeContext


DEFAULT_RUNTIME_PROFILE: dict[str, Any] = {
    "name": "default",
    "device": {
        "type": "auto",
        "ids": None,
    },
    "precision": {
        "amp": False,
    },
    "dataloader": {
        "num_workers": None,
        "pin_memory": False,
        "persistent_workers": False,
        "non_blocking": False,
    },
    "profiling": {
        "enabled": True,
        "record_timing": True,
        "record_memory": False,
    },
    "distributed": {
        "backend": "nccl",
        "find_unused_parameters": False,
    },
}


def load_runtime_profile(path_or_cfg: str | Path | dict[str, Any] | None) -> dict[str, Any]:
    """Load and normalize a runtime profile from YAML or a dict."""

    profile = _deep_copy(DEFAULT_RUNTIME_PROFILE)
    if path_or_cfg is None:
        return profile
    if isinstance(path_or_cfg, (str, Path)):
        raw = load_yaml_config(path_or_cfg)
    elif isinstance(path_or_cfg, dict):
        raw = path_or_cfg
    else:
        raise TypeError("Runtime profile must be a path, dict, or None.")
    _deep_update(profile, _normalize_runtime_payload(raw))
    return profile


def _normalize_runtime_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy and new runtime schemas into one internal shape."""

    normalized = _deep_copy(DEFAULT_RUNTIME_PROFILE)
    if "name" in raw:
        normalized["name"] = raw["name"]

    device_payload = raw.get("device", DEFAULT_RUNTIME_PROFILE["device"])
    normalized["device"] = _normalize_device_payload(device_payload)

    precision_payload = raw.get("precision", {})
    if not isinstance(precision_payload, dict):
        raise TypeError("runtime precision must be a mapping when provided.")
    normalized["precision"] = {
        "amp": bool(precision_payload.get("amp", raw.get("amp", False))),
    }

    dataloader_payload = raw.get("dataloader", {})
    if dataloader_payload is not None:
        if not isinstance(dataloader_payload, dict):
            raise TypeError("runtime dataloader must be a mapping when provided.")
        _deep_update(
            normalized["dataloader"],
            {
                key: value
                for key, value in dataloader_payload.items()
                if key != "batch_size"
            },
        )

    profiling_payload = raw.get("profiling", {})
    if profiling_payload is not None:
        if not isinstance(profiling_payload, dict):
            raise TypeError("runtime profiling must be a mapping when provided.")
        _deep_update(normalized["profiling"], profiling_payload)

    distributed_payload = raw.get("distributed", {})
    if distributed_payload is not None:
        if not isinstance(distributed_payload, dict):
            raise TypeError("runtime distributed must be a mapping when provided.")
        _deep_update(
            normalized["distributed"],
            {
                key: value
                for key, value in distributed_payload.items()
                if key != "enabled"
            },
        )

    return normalized


def _normalize_device_payload(payload: Any) -> dict[str, Any]:
    """Normalize runtime device payloads from legacy and new schema shapes."""

    if isinstance(payload, str):
        if payload in {"auto", "cpu"}:
            return {"type": payload, "ids": None}
        if payload == "cuda":
            return {"type": "cuda", "ids": [0]}
        if payload.startswith("cuda:"):
            index = int(payload.split(":", maxsplit=1)[1])
            if index < 0:
                raise ValueError("CUDA device ids must be non-negative.")
            return {"type": "cuda", "ids": [index]}
        raise ValueError(f"Unsupported runtime device request: {payload}")

    if not isinstance(payload, dict):
        raise TypeError("runtime device must be a string or mapping.")

    device_type = str(payload.get("type", "auto"))
    ids = payload.get("ids")
    if device_type == "cuda":
        if ids is None:
            normalized_ids: list[int] | None = [0]
        else:
            if not isinstance(ids, list) or not ids:
                raise ValueError("runtime device.ids must be a non-empty list when provided.")
            normalized_ids = []
            for raw_id in ids:
                gpu_id = int(raw_id)
                if gpu_id < 0:
                    raise ValueError("runtime device.ids entries must be non-negative.")
                normalized_ids.append(gpu_id)
            if len(set(normalized_ids)) != len(normalized_ids):
                raise ValueError("runtime device.ids entries must be unique.")
        return {"type": "cuda", "ids": normalized_ids}

    if device_type in {"cpu", "auto"}:
        if ids is not None:
            raise ValueError(f"runtime device.ids is only valid for device.type=cuda, got {device_type}.")
        return {"type": device_type, "ids": None}

    raise ValueError(f"Unsupported runtime device type: {device_type}")


def resolve_dataloader_runtime(runtime_cfg: dict[str, Any], defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return normalized DataLoader runtime kwargs with optional defaults."""

    resolved = dict(defaults or {})
    dataloader_cfg = runtime_cfg.get("dataloader", {})
    if isinstance(dataloader_cfg, dict):
        resolved.update(
            {
                key: value
                for key, value in dataloader_cfg.items()
                if value is not None and key != "batch_size"
            }
        )

    if resolved.get("num_workers", 0) == 0:
        resolved["persistent_workers"] = False
    return resolved


def configure_trainable_runtime(
    model: object,
    device: torch.device,
    amp_enabled: bool,
    distributed_context: DistributedRuntimeContext | None = None,
) -> None:
    """Attach runtime device/AMP hints to a trainable model."""

    context = distributed_context or DistributedRuntimeContext()
    if isinstance(model, nn.Module):
        model.to(device)
    setattr(model, "runtime_device", device)
    effective_amp = bool(amp_enabled and device.type == "cuda")
    setattr(model, "amp_enabled", effective_amp)
    setattr(model, "grad_scaler", torch.amp.GradScaler("cuda", enabled=effective_amp))
    setattr(model, "distributed_context", context)
    distributed_training_enabled = False
    if context.enabled and context.world_size > 1:
        distributed_training_enabled = _wrap_trainable_modules_for_distributed(model, device, context)
    setattr(model, "distributed_training_enabled", distributed_training_enabled)
    if effective_amp or device.type != "cpu" or distributed_training_enabled:
        _wrap_fit_for_runtime(model)


def autocast_context(device: torch.device, amp_enabled: bool):
    """Return the appropriate autocast context or a no-op context."""

    if amp_enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def move_sample_to_device(sample: Sample, device: torch.device, non_blocking: bool = False) -> Sample:
    """Move tensor-valued sample fields onto the target device in place."""

    if isinstance(sample.image, torch.Tensor):
        sample.image = sample.image.to(device, non_blocking=non_blocking)
    if isinstance(sample.mask, torch.Tensor):
        sample.mask = sample.mask.to(device, non_blocking=non_blocking)
    if isinstance(sample.reference, torch.Tensor):
        sample.reference = sample.reference.to(device, non_blocking=non_blocking)
    if sample.views:
        sample.views = {
            key: value.to(device, non_blocking=non_blocking) if isinstance(value, torch.Tensor) else value
            for key, value in sample.views.items()
        }
    return sample


def move_samples_to_device(
    samples: Sequence[Sample],
    device: torch.device,
    non_blocking: bool = False,
) -> list[Sample]:
    """Move a batch of samples onto the target device in place."""

    return [move_sample_to_device(sample, device, non_blocking=non_blocking) for sample in samples]


class RuntimeRepresentationAdapter:
    """Move samples to the runtime device before delegating to a representation model."""

    def __init__(self, representation: Any, device: torch.device, non_blocking: bool = False) -> None:
        self.representation = representation
        self.device = device
        self.non_blocking = non_blocking
        self._move_representation_to_device()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.representation, name)

    def __call__(self, sample: Sample) -> dict[str, Any]:
        move_sample_to_device(sample, self.device, non_blocking=self.non_blocking)
        return self.representation(sample)

    def encode_sample(self, sample: Sample) -> Any:
        move_sample_to_device(sample, self.device, non_blocking=self.non_blocking)
        if hasattr(self.representation, "encode_sample"):
            return self.representation.encode_sample(sample)
        if hasattr(self.representation, "encode_batch"):
            batch = self.representation.encode_batch([sample])
            outputs = batch.unbind()
            if len(outputs) != 1:
                raise ValueError(
                    f"{type(self.representation).__name__}.encode_sample expected exactly one output for one input sample, "
                    f"got batch_size={batch.batch_size}."
                )
            return outputs[0]
        raise TypeError(f"{type(self.representation).__name__} does not expose encode_sample() or encode_batch().")

    def encode_batch(self, samples: Sequence[Sample]) -> Any:
        moved_samples = move_samples_to_device(samples, self.device, non_blocking=self.non_blocking)
        if hasattr(self.representation, "encode_batch"):
            return self.representation.encode_batch(moved_samples)
        if not moved_samples:
            raise ValueError("encode_batch() requires at least one sample.")
        outputs = [self.encode_sample(sample) for sample in moved_samples]
        if not outputs:
            raise ValueError("encode_batch() requires at least one sample.")
        first_output = outputs[0]
        if not hasattr(first_output, "provenance"):
            raise TypeError(f"{type(self.representation).__name__} encode_sample() must return RepresentationOutput values.")
        batch_tensor = torch.stack([output.tensor for output in outputs], dim=0)
        from adrf.representation.contracts import RepresentationBatch

        return RepresentationBatch(
            tensor=batch_tensor,
            space=first_output.space,
            spatial_shape=first_output.spatial_shape,
            feature_dim=first_output.feature_dim,
            batch_size=len(outputs),
            sample_ids=tuple(output.sample_id for output in outputs),
            requires_grad=bool(batch_tensor.requires_grad),
            device=str(batch_tensor.device),
            dtype=str(batch_tensor.dtype),
            provenance=first_output.provenance,
        )

    def _move_representation_to_device(self) -> None:
        """Move known representation modules onto the runtime device."""

        if isinstance(self.representation, nn.Module):
            self.representation.to(self.device)
            return

        backbone = getattr(self.representation, "backbone", None)
        if isinstance(backbone, nn.Module):
            backbone.to(self.device)


def _deep_copy(payload: dict[str, Any]) -> dict[str, Any]:
    """Create a recursive copy of nested dictionaries."""

    result: dict[str, Any] = {}
    for key, value in payload.items():
        result[key] = _deep_copy(value) if isinstance(value, dict) else value
    return result


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> None:
    """Recursively merge one dictionary into another."""

    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _wrap_fit_for_runtime(model: object) -> None:
    """Patch supported trainable models so fit() respects runtime device and AMP."""

    if getattr(model, "_adrf_runtime_wrapped", False):
        return

    if hasattr(model, "encoder") and hasattr(model, "decoder") and hasattr(model, "_forward_impl"):
        model.fit = _make_autoencoder_fit(model)  # type: ignore[method-assign]
    elif hasattr(model, "conditional_denoiser") and hasattr(model, "_prepare_reference_tensor"):
        model.fit = _make_reference_diffusion_fit(model)  # type: ignore[method-assign]
    elif hasattr(model, "conditional_model") and hasattr(model, "_prepare_reference_tensor"):
        model.fit = _make_reference_basic_fit(model)  # type: ignore[method-assign]
    elif hasattr(model, "denoiser") and hasattr(model, "_sample_noisy_inputs"):
        model.fit = _make_diffusion_fit(model)  # type: ignore[method-assign]

    setattr(model, "_adrf_runtime_wrapped", True)


def _wrap_trainable_modules_for_distributed(
    model: object,
    device: torch.device,
    distributed_context: DistributedRuntimeContext,
) -> bool:
    """Wrap trainable submodules with DDP when requested."""

    if getattr(model, "_adrf_distributed_wrapped", False):
        return bool(getattr(model, "distributed_training_enabled", False))

    wrapped_any = False
    for module_name in _distributed_trainable_module_names(model):
        module = getattr(model, module_name, None)
        if isinstance(module, nn.Module):
            setattr(
                model,
                module_name,
                _wrap_module_for_distributed(module, device, distributed_context),
            )
            wrapped_any = True

    setattr(model, "_adrf_distributed_wrapped", True)
    return wrapped_any


def _distributed_trainable_module_names(model: object) -> tuple[str, ...]:
    """Return the trainable submodules that should be wrapped for DDP."""

    if hasattr(model, "encoder") and hasattr(model, "decoder"):
        return ("encoder", "decoder")
    if hasattr(model, "conditional_denoiser"):
        return ("conditional_denoiser",)
    if hasattr(model, "conditional_model"):
        return ("conditional_model",)
    if hasattr(model, "denoiser"):
        return ("denoiser",)
    return ()


def _wrap_module_for_distributed(
    module: nn.Module,
    device: torch.device,
    distributed_context: DistributedRuntimeContext,
) -> DistributedDataParallel:
    """Wrap one module with DDP using CPU or CUDA-appropriate arguments."""

    kwargs: dict[str, Any] = {
        "find_unused_parameters": distributed_context.find_unused_parameters,
    }
    if device.type == "cuda":
        device_index = int(device.index or 0)
        kwargs["device_ids"] = [device_index]
        kwargs["output_device"] = device_index
    return DistributedDataParallel(module, **kwargs)


def _wrap_diffusers_adapter_for_distributed(model: Any) -> None:
    """Wrap a lazily-instantiated diffusers adapter model when distributed is enabled."""

    distributed_context = getattr(model, "distributed_context", DistributedRuntimeContext())
    if not distributed_context.enabled or distributed_context.world_size <= 1:
        return
    adapter = getattr(model, "diffusers_adapter", None)
    if adapter is None or not hasattr(adapter, "model"):
        return
    if isinstance(adapter.model, DistributedDataParallel):
        return
    adapter.model = _wrap_module_for_distributed(
        adapter.model,
        getattr(model, "runtime_device", torch.device("cpu")),
        distributed_context,
    )


def _autocast_context(model: object):
    """Return the correct autocast context for the model runtime."""

    runtime_device = getattr(model, "runtime_device", torch.device("cpu"))
    amp_enabled = bool(getattr(model, "amp_enabled", False))
    if amp_enabled and runtime_device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _optimizer_step(model: object, optimizer: torch.optim.Optimizer, loss: torch.Tensor) -> None:
    """Run one optimizer step with or without GradScaler."""

    scaler = getattr(model, "grad_scaler", None)
    optimizer.zero_grad()
    if scaler is not None and scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()


def _make_autoencoder_fit(model: Any):
    """Create a runtime-aware fit function for AutoEncoderNormality."""

    def fit(representations, samples=None):
        del samples
        tensors = [model.require_representation_tensor(representation).float() for representation in representations]
        if not tensors:
            raise ValueError("AutoEncoderNormality.fit requires at least one representation.")

        train_batch = torch.stack(tensors, dim=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
        model.train()
        for _ in range(model.epochs):
            permutation = torch.randperm(train_batch.shape[0], device=train_batch.device)
            shuffled = train_batch[permutation]
            for start in range(0, shuffled.shape[0], model.batch_size):
                batch = shuffled[start : start + model.batch_size]
                with _autocast_context(model):
                    projection, reconstruction = model._forward_impl(batch)
                    del projection
                    loss = functional.mse_loss(reconstruction, batch)
                _optimizer_step(model, optimizer, loss)
                model.last_fit_loss = float(loss.detach().cpu().item())
        model.eval()

    return fit


def _make_diffusion_fit(model: Any):
    """Create a runtime-aware fit function for DiffusionBasicNormality."""

    def fit(representations, samples=None):
        del samples
        tensors = [model.require_representation_tensor(representation).float() for representation in representations]
        if not tensors:
            raise ValueError("DiffusionBasicNormality.fit requires at least one representation.")

        train_batch = torch.stack(tensors, dim=0)
        if getattr(model, "backend", "legacy") == "diffusers":
            model._ensure_diffusers_backend(sample_size=int(train_batch.shape[-1]))
            optimizer = torch.optim.Adam(model.diffusers_adapter.parameters(), lr=model.learning_rate)
            model.diffusers_adapter.train()
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
        model.train()
        for _ in range(model.epochs):
            permutation = torch.randperm(train_batch.shape[0], device=train_batch.device)
            shuffled = train_batch[permutation]
            for start in range(0, shuffled.shape[0], model.batch_size):
                clean_batch = shuffled[start : start + model.batch_size]
                with _autocast_context(model):
                    if getattr(model, "backend", "legacy") == "diffusers":
                        loss, _ = model.diffusers_adapter.forward_train_step(clean_batch)
                    else:
                        sampled = model._sample_noisy_inputs(clean_batch)
                        if len(sampled) == 4:
                            noisy_batch, target_noise, timesteps, noise_scales = sampled
                            predicted_noise = model.denoiser(noisy_batch, timesteps, noise_scales)
                        else:
                            noisy_batch, target_noise, timesteps = sampled
                            predicted_noise = model.denoiser(noisy_batch, timesteps)
                        loss = functional.mse_loss(predicted_noise, target_noise)
                _optimizer_step(model, optimizer, loss)
                model.last_fit_loss = float(loss.detach().cpu().item())
        model.eval()

    return fit


def _make_reference_basic_fit(model: Any):
    """Create a runtime-aware fit function for ReferenceBasicNormality."""

    def fit(representations, samples=None):
        if samples is None:
            raise ValueError("ReferenceBasicNormality.fit requires samples with reference inputs.")

        paired_tensors = [
            (
                model.require_representation_tensor(representation).float(),
                model._prepare_reference_tensor(sample, representation).float(),
            )
            for representation, sample in zip(representations, samples, strict=True)
        ]
        if not paired_tensors:
            raise ValueError("ReferenceBasicNormality.fit requires at least one representation.")

        image_batch = torch.stack([image for image, _ in paired_tensors], dim=0)
        reference_batch = torch.stack([reference for _, reference in paired_tensors], dim=0)
        conditional_batch = torch.cat([image_batch, reference_batch], dim=1)

        optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
        model.train()
        for _ in range(model.epochs):
            permutation = torch.randperm(conditional_batch.shape[0], device=conditional_batch.device)
            shuffled_inputs = conditional_batch[permutation]
            shuffled_targets = image_batch[permutation]
            for start in range(0, shuffled_inputs.shape[0], model.batch_size):
                batch_inputs = shuffled_inputs[start : start + model.batch_size]
                batch_targets = shuffled_targets[start : start + model.batch_size]
                with _autocast_context(model):
                    projection = model.conditional_model(batch_inputs)
                    loss = functional.mse_loss(projection, batch_targets)
                _optimizer_step(model, optimizer, loss)
                model.last_fit_loss = float(loss.detach().cpu().item())
        model.eval()

    return fit


def _make_reference_diffusion_fit(model: Any):
    """Create a runtime-aware fit function for ReferenceDiffusionBasicNormality."""

    def fit(representations, samples=None):
        if samples is None:
            raise ValueError("ReferenceDiffusionBasicNormality.fit requires samples with reference inputs.")

        paired_tensors = [
            (
                model.require_representation_tensor(representation).float(),
                model._prepare_reference_tensor(sample, representation).float(),
            )
            for representation, sample in zip(representations, samples, strict=True)
        ]
        if not paired_tensors:
            raise ValueError("ReferenceDiffusionBasicNormality.fit requires at least one representation.")

        image_batch = torch.stack([image for image, _ in paired_tensors], dim=0)
        reference_batch = torch.stack([reference for _, reference in paired_tensors], dim=0)

        optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
        model.train()
        for _ in range(model.epochs):
            permutation = torch.randperm(image_batch.shape[0], device=image_batch.device)
            shuffled_images = image_batch[permutation]
            shuffled_references = reference_batch[permutation]
            for start in range(0, shuffled_images.shape[0], model.batch_size):
                clean_batch = shuffled_images[start : start + model.batch_size]
                reference_slice = shuffled_references[start : start + model.batch_size]
                with _autocast_context(model):
                    noisy_batch, target_noise, timesteps = model._sample_noisy_inputs(clean_batch)
                    conditional_inputs = torch.cat([noisy_batch, reference_slice], dim=1)
                    predicted_noise = model.conditional_denoiser(conditional_inputs, timesteps)
                    loss = functional.mse_loss(predicted_noise, target_noise)
                _optimizer_step(model, optimizer, loss)
                model.last_fit_loss = float(loss.detach().cpu().item())
        model.eval()

    return fit
