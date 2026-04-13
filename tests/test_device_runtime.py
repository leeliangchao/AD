"""Tests for runtime device resolution helpers."""

from unittest.mock import patch

import torch

from adrf.utils.device import resolve_device


def test_resolve_device_auto_falls_back_to_cpu_when_cuda_unavailable() -> None:
    """`device: auto` should resolve to CPU when CUDA is unavailable."""

    with patch("torch.cuda.is_available", return_value=False):
        device, info = resolve_device({"name": "smoke", "device": "auto", "amp": True})

    assert device.type == "cpu"
    assert info["requested_device"] == "auto"
    assert info["actual_device"] == "cpu"
    assert info["cuda_available"] is False
    assert info["amp_enabled"] is False


def test_resolve_device_honors_cuda_when_available() -> None:
    """Explicit CUDA requests should resolve to a CUDA device when available."""

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_name", return_value="Fake GPU"),
    ):
        device, info = resolve_device({"name": "real", "device": "cuda", "amp": True})

    assert device.type == "cuda"
    assert info["actual_device"] == "cuda"
    assert info["device_name"] == "Fake GPU"
    assert info["amp_enabled"] is True

