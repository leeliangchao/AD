"""Benchmark suite configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from adrf.utils.config import load_yaml_config


@dataclass(slots=True)
class BenchmarkSuite:
    """A named sequence of experiment configs to run sequentially."""

    name: str
    experiments: list[Path]
    source_path: Path
    continue_on_error: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchmarkSuite":
        """Load a benchmark suite from YAML and resolve experiment paths."""

        source_path = Path(path).resolve()
        config = load_yaml_config(source_path)
        raw_experiments = config.get("experiments", [])
        if not isinstance(raw_experiments, list) or not raw_experiments:
            raise ValueError("Benchmark suite config must define a non-empty 'experiments' list.")
        experiments = [
            (source_path.parent / Path(experiment)).resolve()
            for experiment in raw_experiments
        ]
        return cls(
            name=str(config.get("name", source_path.stem)),
            experiments=experiments,
            source_path=source_path,
            continue_on_error=bool(config.get("continue_on_error", True)),
        )

