"""Launch helpers that choose between in-process and CLI-distributed execution."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

from adrf.runner.experiment_runner import ExperimentRunner


def run_experiment_with_runtime_launch(
    config: str | Path | dict[str, Any],
    *,
    output_root: str | Path = "outputs/runs",
    run_name: str | None = None,
) -> tuple[dict[str, Any], Path | None, dict[str, Any]]:
    """Run one experiment directly or through the CLI launcher when distributed launch is required."""

    output_root_path = Path(output_root).resolve()
    output_root_path.mkdir(parents=True, exist_ok=True)

    config_path = _materialize_config_path(config, output_root_path=output_root_path)
    from adrf.cli.main import resolve_distributed_launch_plan

    launch_plan = resolve_distributed_launch_plan(config_path)
    if launch_plan is None:
        runner = ExperimentRunner(config, output_root=output_root_path, run_name=run_name)
        results = runner.run()
        return results, runner.run_dir, _load_run_info(runner.run_dir)

    before = _run_directories(output_root_path)
    completed = subprocess.run(
        [sys.executable, "-m", "adrf.cli.main", str(config_path)],
        cwd=output_root_path.parent.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        raise RuntimeError(stderr or stdout or "Distributed experiment launch failed.")
    results = json.loads(completed.stdout)
    run_dir = _detect_new_run_dir(output_root_path, before)
    return results, run_dir, _load_run_info(run_dir)


def _materialize_config_path(config: str | Path | dict[str, Any], *, output_root_path: Path) -> Path:
    if isinstance(config, (str, Path)):
        return Path(config).resolve()

    launch_dir = output_root_path.parent / ".launch_configs"
    launch_dir.mkdir(parents=True, exist_ok=True)
    config_name = str(config.get("name", "experiment"))
    config_path = launch_dir / f"{config_name}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path.resolve()


def _run_directories(output_root_path: Path) -> set[Path]:
    return {path.resolve() for path in output_root_path.iterdir() if path.is_dir()} if output_root_path.exists() else set()


def _detect_new_run_dir(output_root_path: Path, before: set[Path]) -> Path | None:
    after = _run_directories(output_root_path)
    new_dirs = sorted(after - before, key=lambda path: path.stat().st_mtime)
    if new_dirs:
        return new_dirs[-1]
    existing_dirs = sorted(after, key=lambda path: path.stat().st_mtime)
    return existing_dirs[-1] if existing_dirs else None


def _load_run_info(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    run_info_path = run_dir / "run_info.json"
    if not run_info_path.exists():
        return {}
    return json.loads(run_info_path.read_text(encoding="utf-8"))
