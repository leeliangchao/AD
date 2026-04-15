"""Single-experiment report export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adrf.utils.config import load_yaml_config


def export_experiment_report(run_dir: str | Path) -> Path:
    """Generate a markdown report for one completed run directory."""

    run_path = Path(run_dir).resolve()
    run_info = _load_json(run_path / "run_info.json")
    metrics_payload = _load_json(run_path / "metrics.json")
    config_snapshot = load_yaml_config(run_path / "config_snapshot.yaml")

    lines = [
        f"# {run_info.get('run_name', run_path.name)}",
        "",
        f"- Status: {run_info.get('status', 'unknown')}",
        f"- Run Path: `{run_path}`",
    ]
    if run_info.get("started_at"):
        lines.append(f"- Started At: {run_info['started_at']}")
    if run_info.get("finished_at"):
        lines.append(f"- Finished At: {run_info['finished_at']}")

    runtime = run_info.get("runtime", {})
    if isinstance(runtime, dict) and runtime:
        lines.extend(["", "## Runtime", ""])
        for label, key in (
            ("Runtime Profile", "profile_name"),
            ("Requested Device", "requested_device"),
            ("Actual Device", "actual_device"),
            ("Device Name", "device_name"),
            ("AMP Enabled", "amp_enabled"),
            ("Train Time (s)", "train_time_s"),
            ("Eval Time (s)", "eval_time_s"),
            ("Total Time (s)", "total_time_s"),
            ("Peak Memory (bytes)", "peak_memory_bytes"),
        ):
            value = runtime.get(key)
            if value is not None:
                lines.append(f"- {label}: `{value}`")

    components = run_info.get("components", {})
    if isinstance(components, dict) and components:
        lines.extend(["", "## Components", ""])
        for key, value in components.items():
            lines.append(f"- {key}: `{value}`")

    latest_metrics = metrics_payload.get("latest", {})
    if latest_metrics:
        lines.extend(["", "## Metrics", "", "| Metric | Value |", "| --- | ---: |"])
        for key, value in sorted(latest_metrics.items()):
            lines.append(f"| `{key}` | `{value}` |")

    lines.extend(["", "## Config", "", "```yaml"])
    lines.append(_yaml_dump(config_snapshot).rstrip())
    lines.extend(["```", ""])

    report_path = run_path / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def find_latest_run_dir(base_dir: str | Path = "outputs/runs") -> Path:
    """Return the most recently modified run directory."""

    root = Path(base_dir)
    candidates = [path for path in root.iterdir() if path.is_dir()] if root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {root}.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file into a dictionary."""

    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _yaml_dump(payload: dict[str, Any]) -> str:
    """Serialize a YAML-like snapshot without adding a dependency here."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to export experiment reports.") from exc
    return yaml.safe_dump(payload, sort_keys=False)
