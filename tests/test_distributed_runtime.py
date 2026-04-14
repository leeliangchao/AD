"""Tests for distributed runtime helpers and rank-aware runner behavior."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from adrf.normality.autoencoder import AutoEncoderNormality
from adrf.runner.experiment_runner import ExperimentRunner
from adrf.utils.distributed import DistributedRuntimeContext, resolve_distributed_context
from adrf.utils.runtime import configure_trainable_runtime


class _FakeDDP(nn.Module):
    """A minimal stand-in for DistributedDataParallel in unit tests."""

    def __init__(self, module: nn.Module, **kwargs: object) -> None:
        super().__init__()
        self.module = module
        self.kwargs = kwargs

    def forward(self, *args: object, **kwargs: object) -> object:
        return self.module(*args, **kwargs)


def test_resolve_distributed_context_reads_torchrun_environment(monkeypatch) -> None:
    """Distributed runtime settings should be derived from torchrun environment variables."""

    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")

    context = resolve_distributed_context(
        {
            "device": {"type": "cpu"},
            "distributed": {"backend": "gloo"},
        }
    )

    assert context.enabled is True
    assert context.backend == "gloo"
    assert context.rank == 2
    assert context.local_rank == 1
    assert context.world_size == 4
    assert context.is_primary is False


def test_resolve_distributed_context_enables_multi_gpu_from_device_ids() -> None:
    """Multiple CUDA device ids should automatically imply distributed execution."""

    context = resolve_distributed_context(
        {
            "device": {"type": "cuda", "ids": [0, 3]},
            "distributed": {"backend": "nccl"},
        }
    )

    assert context.enabled is True
    assert context.world_size == 2
    assert context.rank == 0
    assert context.local_rank == 0
    assert context.is_primary is True


def test_configure_trainable_runtime_wraps_trainable_submodules_for_ddp() -> None:
    """Distributed runtime should wrap trainable submodules instead of the whole normality object."""

    model = AutoEncoderNormality(input_channels=3, hidden_channels=4, latent_channels=8)
    context = DistributedRuntimeContext(enabled=True, backend="gloo", rank=0, local_rank=0, world_size=2)

    with patch("adrf.utils.runtime.DistributedDataParallel", _FakeDDP):
        configure_trainable_runtime(model, device=torch.device("cpu"), amp_enabled=False, distributed_context=context)

    assert isinstance(model.encoder, _FakeDDP)
    assert isinstance(model.decoder, _FakeDDP)


def test_experiment_runner_skips_local_run_artifacts_on_non_primary_rank(tmp_path: Path, monkeypatch) -> None:
    """Non-primary distributed ranks should not create the main local run directory."""

    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    config = {
        "name": "distributed_non_primary_demo",
        "runtime_config": {
            "device": {"type": "cpu"},
            "precision": {"amp": False},
            "distributed": {
                "backend": "gloo",
            },
            "dataloader": {
                "batch_size": 2,
                "num_workers": 0,
            },
        },
        "datamodule": {
            "name": "mvtec_single_class",
            "params": {
                "root": str(Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mvtec"),
                "category": "bottle",
                "reference_index": 0,
                "image_size": [32, 32],
                "batch_size": 2,
                "num_workers": 0,
                "normalize": False,
            },
        },
        "representation": {"name": "pixel", "params": {}},
        "normality": {
            "name": "feature_memory",
            "params": {},
        },
        "evidence": {"name": "feature_distance", "params": {"aggregator": "max"}},
        "evaluator": {"name": "basic_ad", "params": {}},
        "protocol": {"name": "one_class", "params": {}},
    }

    runner = ExperimentRunner(config, output_root=tmp_path / "runs")
    runner._start_logged_run()

    assert runner.run_dir is None
    assert list((tmp_path / "runs").glob("*")) == []
