"""CLI entrypoint for running a single experiment config."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import sys

from adrf.runner.experiment_runner import ExperimentRunner
from adrf.utils.config import load_yaml_config
from adrf.utils.runtime import load_runtime_profile


@dataclass(slots=True)
class DistributedLaunchPlan:
    """Launch metadata for a torchrun-based experiment re-exec."""

    config_path: Path
    visible_device_ids: list[int]
    nproc_per_node: int


def main(argv: Sequence[str] | None = None) -> int:
    """Run one experiment config and print the resulting metrics."""

    parser = argparse.ArgumentParser(description="Run one AD research experiment.")
    parser.add_argument(
        "config",
        nargs="?",
        default=Path("configs/experiment/feature_baseline.yaml"),
        type=Path,
        help="Path to the experiment config file.",
    )
    args = parser.parse_args(argv)

    launch_plan = resolve_distributed_launch_plan(args.config)
    if launch_plan is not None:
        return launch_distributed_experiment(launch_plan)

    results = ExperimentRunner(args.config).run()
    if os.environ.get("RANK") in {None, "0"}:
        print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def resolve_distributed_launch_plan(config: str | Path) -> DistributedLaunchPlan | None:
    """Return a torchrun launch plan when the runtime requests multi-GPU execution."""

    if os.environ.get("LOCAL_RANK") is not None:
        return None

    config_path = Path(config).resolve()
    config_payload = load_yaml_config(config_path)
    runtime_candidate = config_payload.get("runtime_config")
    runtime_profile = load_runtime_profile(_resolve_runtime_candidate(config_path, runtime_candidate))
    device_cfg = runtime_profile.get("device", {})
    if not isinstance(device_cfg, dict):
        return None
    if str(device_cfg.get("type", "auto")) != "cuda":
        return None
    device_ids = device_cfg.get("ids")
    if not isinstance(device_ids, list) or len(device_ids) <= 1:
        return None

    return DistributedLaunchPlan(
        config_path=config_path,
        visible_device_ids=[int(device_id) for device_id in device_ids],
        nproc_per_node=len(device_ids),
    )


def launch_distributed_experiment(plan: DistributedLaunchPlan) -> int:
    """Re-launch the experiment under torch.distributed.run."""

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(device_id) for device_id in plan.visible_device_ids)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(plan.nproc_per_node),
        "-m",
        "adrf.cli.main",
        str(plan.config_path),
    ]
    completed = subprocess.run(command, env=env, check=False)
    return int(completed.returncode)


def _resolve_runtime_candidate(
    config_path: Path,
    runtime_candidate: object,
) -> str | Path | dict[str, object] | None:
    """Resolve a runtime config reference relative to the repo or config file."""

    if runtime_candidate is None or isinstance(runtime_candidate, dict):
        return runtime_candidate
    runtime_path = Path(runtime_candidate)
    if runtime_path.is_absolute():
        return runtime_path
    repo_root = Path(__file__).resolve().parents[3]
    repo_relative = (repo_root / runtime_path).resolve()
    if repo_relative.exists():
        return repo_relative
    return (config_path.parent / runtime_path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
