"""Run one experiment config selected by a single editable YAML filename."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from adrf.runner.experiment_runner import ExperimentRunner


# Edit this filename when you want to right-click run a different experiment config.
CONFIG_NAME = "diffusion_baseline_bottle_real.yaml"


def resolve_config_path(config_name: str = CONFIG_NAME) -> Path:
    """Resolve the configured YAML file to an absolute path inside the repository."""

    candidate = Path(config_name)
    if candidate.is_absolute():
        resolved = candidate
    elif candidate.parent == Path("."):
        resolved = (PROJECT_ROOT / "configs" / "experiment" / candidate).resolve()
    else:
        resolved = (PROJECT_ROOT / candidate).resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"Experiment config not found: {resolved}")
    return resolved


def main() -> int:
    """Run the configured experiment and print the resulting metrics payload."""

    config_path = resolve_config_path()
    results = ExperimentRunner(config_path, output_root=PROJECT_ROOT / "outputs" / "runs").run()
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
