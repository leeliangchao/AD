"""Device resolution helpers for runtime-aware experiment execution."""

from __future__ import annotations

import warnings
from typing import Any

import torch


def resolve_device(
    runtime_cfg: dict[str, Any],
    distributed_context: Any | None = None,
) -> tuple[torch.device, dict[str, Any]]:
    """Resolve the requested runtime device and return device metadata."""

    requested_device = runtime_cfg.get("device", {"type": "auto", "ids": None})
    if not isinstance(requested_device, dict):
        raise TypeError("runtime device must be normalized to a mapping before resolve_device().")
    requested_device_type = str(requested_device.get("type", "auto"))
    requested_device_ids = requested_device.get("ids")
    amp_requested = bool((runtime_cfg.get("precision", {}) or {}).get("amp", False))
    distributed_enabled = bool(getattr(distributed_context, "enabled", False))
    distributed_world_size = int(getattr(distributed_context, "world_size", 1))
    distributed_local_rank = int(getattr(distributed_context, "local_rank", 0))

    cuda_available = torch.cuda.is_available()
    actual_index: int | None = None
    if requested_device_type == "auto":
        actual_device = "cuda" if cuda_available else "cpu"
        if actual_device == "cuda" and distributed_enabled and distributed_world_size > 1:
            actual_index = distributed_local_rank
    elif requested_device_type == "cuda":
        if cuda_available:
            actual_device = "cuda"
            actual_index = (
                distributed_local_rank
                if distributed_enabled and distributed_world_size > 1
                else _single_device_index(requested_device_ids)
            )
        else:
            warnings.warn(
                "CUDA was requested but is not available; falling back to CPU.",
                stacklevel=2,
            )
            actual_device = "cpu"
    elif requested_device_type == "cpu":
        actual_device = "cpu"
    else:
        raise ValueError(f"Unsupported runtime device request: {requested_device}")

    if actual_device == "cuda":
        visible_device_count = torch.cuda.device_count()
        validation_index = int(actual_index or 0)
        if validation_index < 0 or validation_index >= visible_device_count:
            raise ValueError(
                f"Requested CUDA device index {validation_index} is out of range for "
                f"{visible_device_count} visible device(s)."
            )
        device = torch.device(f"cuda:{validation_index}") if actual_index is not None else torch.device("cuda")
        actual_device_name = f"cuda:{validation_index}" if actual_index is not None else "cuda"
        device_name = torch.cuda.get_device_name(validation_index)
    else:
        device = torch.device(actual_device)
        actual_device_name = actual_device
        device_name = "cpu"

    amp_enabled = bool(amp_requested and device.type == "cuda")
    info = {
        "requested_device": requested_device,
        "actual_device": actual_device_name,
        "device_type": device.type,
        "device_name": device_name,
        "device_index": actual_index,
        "device_ids": list(requested_device_ids) if isinstance(requested_device_ids, list) else None,
        "cuda_available": cuda_available,
        "amp_enabled": amp_enabled,
    }
    return device, info


def _single_device_index(device_ids: Any) -> int | None:
    """Return the single requested device index for non-distributed execution."""

    if device_ids is None:
        return None
    if not isinstance(device_ids, list) or not device_ids:
        raise ValueError("runtime device.ids must be a non-empty list when provided.")
    if len(device_ids) > 1:
        raise ValueError(
            "Multiple runtime device ids require distributed launch. "
            "Use the CLI launcher or torchrun so each rank can bind one GPU."
        )
    return int(device_ids[0])
