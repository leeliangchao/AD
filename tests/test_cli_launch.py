"""Tests for runtime-driven CLI launch planning."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from adrf.cli import main as experiment_cli


def _write_runtime(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "name: real_ddp",
                "device:",
                "  type: cuda",
                "  ids: [0, 3]",
                "precision:",
                "  amp: true",
                "distributed:",
                "  backend: nccl",
                "  find_unused_parameters: false",
            ]
        ),
        encoding="utf-8",
    )


def _write_experiment(path: Path, runtime_path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                f"runtime_config: {runtime_path.as_posix()}",
                "datamodule:",
                "  name: mvtec_single_class",
                "  params:",
                "    root: tests/fixtures/mvtec",
                "    category: bottle",
                "    reference_index: 0",
                "    image_size: [32, 32]",
                "    batch_size: 2",
                "    num_workers: 0",
                "    normalize: false",
                "representation:",
                "  name: pixel",
                "  params: {}",
                "normality:",
                "  name: feature_memory",
                "  params: {}",
                "evidence:",
                "  name: feature_distance",
                "  params:",
                "    aggregator: max",
                "evaluator:",
                "  name: basic_ad",
                "  params: {}",
                "protocol:",
                "  name: one_class",
                "  params: {}",
            ]
        ),
        encoding="utf-8",
    )


def test_resolve_distributed_launch_plan_uses_runtime_gpu_ids(tmp_path: Path) -> None:
    """CLI launch planning should derive torchrun world size and visible devices from runtime ids."""

    runtime_path = tmp_path / "real_ddp.yaml"
    config_path = tmp_path / "experiment.yaml"
    _write_runtime(runtime_path)
    _write_experiment(config_path, runtime_path)

    plan = experiment_cli.resolve_distributed_launch_plan(config_path)

    assert plan is not None
    assert plan.visible_device_ids == [0, 3]
    assert plan.nproc_per_node == 2
    assert plan.config_path == config_path.resolve()


def test_main_relaunches_under_torchrun_for_multi_gpu_runtime(tmp_path: Path) -> None:
    """The experiment CLI should auto-relaunch itself under torchrun for multi-GPU runtimes."""

    runtime_path = tmp_path / "real_ddp.yaml"
    config_path = tmp_path / "experiment.yaml"
    _write_runtime(runtime_path)
    _write_experiment(config_path, runtime_path)

    with patch("adrf.cli.main.subprocess.run", return_value=SimpleNamespace(returncode=0)) as run_mock:
        exit_code = experiment_cli.main([str(config_path)])

    assert exit_code == 0
    args, kwargs = run_mock.call_args
    command = args[0]
    assert command[:6] == [
        experiment_cli.sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        "2",
    ]
    assert command[-2:] == ["adrf.cli.main", str(config_path.resolve())]
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0,3"
