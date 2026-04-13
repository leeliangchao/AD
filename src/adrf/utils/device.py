"""Device resolution helpers for runtime-aware experiment execution."""

from __future__ import annotations

import warnings
from typing import Any

import torch


def resolve_device(runtime_cfg: dict[str, Any]) -> tuple[torch.device, dict[str, Any]]:
    """Resolve the requested runtime device and return device metadata."""

    requested_device = str(runtime_cfg.get("device", "auto"))
    amp_requested = bool(runtime_cfg.get("amp", False))

    cuda_available = torch.cuda.is_available()
    if requested_device == "auto":
        actual_device = "cuda" if cuda_available else "cpu"
    elif requested_device == "cuda":
        if cuda_available:
            actual_device = "cuda"
        else:
            warnings.warn(
                "CUDA was requested but is not available; falling back to CPU.",
                stacklevel=2,
            )
            actual_device = "cpu"
    elif requested_device == "cpu":
        actual_device = "cpu"
    else:
        raise ValueError(f"Unsupported runtime device request: {requested_device}")

    device = torch.device(actual_device)
    amp_enabled = bool(amp_requested and device.type == "cuda")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    info = {
        "requested_device": requested_device,
        "actual_device": device.type,
        "device_type": device.type,
        "device_name": device_name,
        "cuda_available": cuda_available,
        "amp_enabled": amp_enabled,
    }
    return device, info

