"""Runtime profile loading and lightweight device helpers."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn

from adrf.core.sample import Sample
from adrf.utils.config import load_yaml_config


DEFAULT_RUNTIME_PROFILE: dict[str, Any] = {
    "name": "default",
    "device": "auto",
    "amp": False,
    "dataloader": {
        "batch_size": None,
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
    _deep_update(profile, raw)
    return profile


def resolve_dataloader_runtime(runtime_cfg: dict[str, Any], defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return normalized DataLoader runtime kwargs with optional defaults."""

    resolved = dict(defaults or {})
    dataloader_cfg = runtime_cfg.get("dataloader", {})
    if isinstance(dataloader_cfg, dict):
        resolved.update({key: value for key, value in dataloader_cfg.items() if value is not None})

    if resolved.get("num_workers", 0) == 0:
        resolved["persistent_workers"] = False
    return resolved


def configure_trainable_runtime(model: object, device: torch.device, amp_enabled: bool) -> None:
    """Attach runtime device/AMP hints to a trainable model."""

    if isinstance(model, nn.Module):
        model.to(device)
    setattr(model, "runtime_device", device)
    effective_amp = bool(amp_enabled and device.type == "cuda")
    setattr(model, "amp_enabled", effective_amp)
    setattr(model, "grad_scaler", torch.amp.GradScaler("cuda", enabled=effective_amp))
    if effective_amp or device.type != "cpu":
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


class RuntimeRepresentationAdapter:
    """Move samples to the runtime device before delegating to a representation model."""

    def __init__(self, representation: Any, device: torch.device, non_blocking: bool = False) -> None:
        self.representation = representation
        self.device = device
        self.non_blocking = non_blocking
        self._move_representation_to_device()

    def __call__(self, sample: Sample) -> dict[str, Any]:
        move_sample_to_device(sample, self.device, non_blocking=self.non_blocking)
        return self.representation(sample)

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
                        noisy_batch, target_noise, timesteps = model._sample_noisy_inputs(clean_batch)
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
