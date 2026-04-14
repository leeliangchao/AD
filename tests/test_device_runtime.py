"""Tests for runtime device resolution helpers."""

from unittest.mock import patch

import torch

from adrf.utils.device import resolve_device


def test_resolve_device_auto_falls_back_to_cpu_when_cuda_unavailable() -> None:
    """`device: auto` should resolve to CPU when CUDA is unavailable."""

    with patch("torch.cuda.is_available", return_value=False):
        device, info = resolve_device(
            {
                "name": "smoke",
                "device": {"type": "auto"},
                "precision": {"amp": True},
            }
    )

    assert device.type == "cpu"
    assert info["requested_device"] == {"type": "auto"}
    assert info["actual_device"] == "cpu"
    assert info["cuda_available"] is False
    assert info["amp_enabled"] is False


def test_resolve_device_honors_cuda_when_available() -> None:
    """Explicit CUDA requests should resolve to a CUDA device when available."""

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=4),
        patch("torch.cuda.get_device_name", return_value="Fake GPU"),
    ):
        device, info = resolve_device(
            {
                "name": "real",
                "device": {"type": "cuda", "ids": [0]},
                "precision": {"amp": True},
            }
        )

    assert device.type == "cuda"
    assert info["actual_device"] == "cuda:0"
    assert info["device_name"] == "Fake GPU"
    assert info["device_ids"] == [0]
    assert info["amp_enabled"] is True


def test_resolve_device_supports_explicit_cuda_index() -> None:
    """An explicit CUDA index should resolve to the requested device."""

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=4),
        patch("torch.cuda.get_device_name", return_value="Fake GPU 1"),
    ):
        device, info = resolve_device(
            {
                "name": "real",
                "device": {"type": "cuda", "ids": [1]},
                "precision": {"amp": True},
            }
        )

    assert str(device) == "cuda:1"
    assert info["requested_device"] == {"type": "cuda", "ids": [1]}
    assert info["actual_device"] == "cuda:1"
    assert info["device_index"] == 1
    assert info["device_name"] == "Fake GPU 1"
    assert info["device_ids"] == [1]
    assert info["amp_enabled"] is True


def test_resolve_device_rejects_out_of_range_cuda_index() -> None:
    """Out-of-range CUDA indices should fail loudly."""

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=1),
    ):
        try:
            resolve_device(
                {
                    "name": "real",
                    "device": {"type": "cuda", "ids": [3]},
                    "precision": {"amp": True},
                }
            )
        except ValueError as exc:
            assert "out of range" in str(exc)
        else:
            raise AssertionError("Expected resolve_device() to reject an invalid CUDA index.")
