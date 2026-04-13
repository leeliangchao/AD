"""Experiment matrix parsing and expansion."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from adrf.ablation.compatibility import filter_valid_combinations
from adrf.utils.config import load_yaml_config


@dataclass(slots=True)
class AblationMatrix:
    """A structured experiment matrix that expands into runnable experiment specs."""

    name: str
    config: dict[str, Any]
    source_path: Path

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AblationMatrix":
        """Load an ablation matrix configuration from YAML."""

        source_path = Path(path).resolve()
        config = load_yaml_config(source_path)
        return cls(name=str(config.get("name", source_path.stem)), config=config, source_path=source_path)

    def expand(self) -> list[dict[str, Any]]:
        """Expand the configured matrix into valid runnable experiment specs."""

        combos, compatibility_rules = self._collect_combos_and_rules()
        valid, _invalid = filter_valid_combinations(combos, compatibility_rules)
        specs: list[dict[str, Any]] = []
        for combo in valid:
            specs.extend(self._expand_seed_specs(combo))
        return specs

    def expand_with_invalid(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Expand the matrix and return both valid specs and rejected combos."""

        combos, compatibility_rules = self._collect_combos_and_rules()
        valid, invalid = filter_valid_combinations(combos, compatibility_rules)
        specs: list[dict[str, Any]] = []
        for combo in valid:
            specs.extend(self._expand_seed_specs(combo))
        return specs, invalid

    def _collect_combos_and_rules(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Resolve the matrix schema into raw combos and normalized compatibility rules."""

        if self._is_paper_schema():
            combos = self._expand_paper_schema()
            rules = self._normalize_compatibility_rules(self.config.get("compatibility", {}))
            return combos, rules

        combos = self._collect_rows() if "rows" in self.config else self._expand_axes()
        return combos, self.config.get("compatibility", {})

    def _expand_axes(self) -> list[dict[str, Any]]:
        """Expand a grid-style matrix config into raw alias combinations."""

        axes = self.config.get("axes")
        if not isinstance(axes, dict):
            raise ValueError("Ablation matrix config must define `rows` or an `axes` mapping.")

        datasets = list(axes.get("datasets", []))
        representations = list(axes.get("representations", []))
        normality = list(axes.get("normality", []))
        evidence = list(axes.get("evidence", []))
        if not all([datasets, representations, normality, evidence]):
            raise ValueError("Ablation matrix axes must define datasets, representations, normality, and evidence.")

        protocol_alias = self._default_alias("protocol")
        evaluator_alias = self._default_alias("evaluator")
        matrix_overrides = self._matrix_overrides()
        combos: list[dict[str, Any]] = []
        for dataset_alias, representation_alias, normality_alias, evidence_alias in product(
            datasets,
            representations,
            normality,
            evidence,
        ):
            combos.append(
                {
                    "dataset": dataset_alias,
                    "representation": representation_alias,
                    "normality": normality_alias,
                    "evidence": evidence_alias,
                    "protocol": protocol_alias,
                    "evaluator": evaluator_alias,
                    "overrides": self._merge_nested_mappings(matrix_overrides, {}),
                }
            )
        return combos

    def _expand_paper_schema(self) -> list[dict[str, Any]]:
        """Expand the paper-style schema into raw alias combinations."""

        dataset_entries = self.config.get("datasets")
        normality_aliases = self.config.get("normality")
        evidence_aliases = self.config.get("evidence")
        representation_map = self.config.get("representation_map", {})
        if not isinstance(dataset_entries, list) or not dataset_entries:
            raise ValueError("Paper ablation schema must define a non-empty `datasets` list.")
        if not isinstance(normality_aliases, list) or not normality_aliases:
            raise ValueError("Paper ablation schema must define a non-empty `normality` list.")
        if not isinstance(evidence_aliases, list) or not evidence_aliases:
            raise ValueError("Paper ablation schema must define a non-empty `evidence` list.")
        if not isinstance(representation_map, dict):
            raise TypeError("Paper ablation schema `representation_map` must be a mapping.")

        protocol_alias = str(self.config.get("protocol", "one_class"))
        evaluator_alias = self._resolve_evaluator_alias()
        matrix_overrides = self._matrix_overrides()
        combos: list[dict[str, Any]] = []
        for dataset_entry, normality_alias, evidence_alias in product(
            dataset_entries,
            normality_aliases,
            evidence_aliases,
        ):
            if not isinstance(dataset_entry, dict) or "name" not in dataset_entry:
                raise TypeError("Each paper-schema dataset entry must be a mapping with a `name` field.")
            dataset_alias = str(dataset_entry["name"])
            if normality_alias not in representation_map:
                raise KeyError(f"Paper ablation schema missing representation_map entry for `{normality_alias}`.")
            combos.append(
                {
                    "dataset": dataset_alias,
                    "representation": str(representation_map[normality_alias]),
                    "normality": str(normality_alias),
                    "evidence": str(evidence_alias),
                    "protocol": protocol_alias,
                    "evaluator": evaluator_alias,
                    "overrides": self._merge_nested_mappings(matrix_overrides, {}),
                }
            )
        return combos

    def _collect_rows(self) -> list[dict[str, Any]]:
        """Collect explicit row combinations from the config."""

        rows = self.config.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError("Ablation matrix `rows` must be a non-empty list.")

        protocol_alias = self._default_alias("protocol")
        evaluator_alias = self._default_alias("evaluator")
        matrix_overrides = self._matrix_overrides()
        combos: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                raise TypeError("Each ablation matrix row must be a mapping.")
            combos.append(
                {
                    "dataset": row["dataset"],
                    "representation": row["representation"],
                    "normality": row["normality"],
                    "evidence": row["evidence"],
                    "protocol": row.get("protocol", protocol_alias),
                    "evaluator": row.get("evaluator", evaluator_alias),
                    "overrides": self._merge_nested_mappings(matrix_overrides, row.get("overrides", {})),
                }
            )
        return combos

    def _build_spec(self, combo: dict[str, Any]) -> dict[str, Any]:
        """Construct a runnable experiment spec from one alias combination."""

        dataset_alias = combo["dataset"]
        representation_alias = combo["representation"]
        normality_alias = combo["normality"]
        evidence_alias = combo["evidence"]
        protocol_alias = combo["protocol"]
        evaluator_alias = combo["evaluator"]
        group_name = "__".join(
            [dataset_alias, representation_alias, normality_alias, evidence_alias]
        )
        seed = combo.get("seed")
        experiment_name = f"{group_name}__seed{seed}" if seed is not None else group_name

        config = {
            "name": experiment_name,
            "datamodule": self._resolve_component("datasets", dataset_alias),
            "representation": self._resolve_component("representations", representation_alias),
            "normality": self._resolve_component("normality", normality_alias),
            "evidence": self._resolve_component("evidence", evidence_alias),
            "protocol": self._resolve_component("protocols", protocol_alias),
            "evaluator": self._resolve_component("evaluators", evaluator_alias),
        }
        overrides = combo.get("overrides", {})
        resolved_runtime_config = self._resolve_runtime_config(overrides)
        if resolved_runtime_config is not None:
            config["runtime_config"] = resolved_runtime_config
        if seed is not None:
            config["seed"] = seed
        if isinstance(overrides, dict) and overrides:
            config["overrides"] = deepcopy(overrides)

        return {
            "name": experiment_name,
            "group_name": group_name,
            "dataset": dataset_alias,
            "representation": representation_alias,
            "normality": normality_alias,
            "evidence": evidence_alias,
            "protocol": protocol_alias,
            "evaluator": evaluator_alias,
            "seed": seed,
            "runtime_config": resolved_runtime_config,
            "overrides": deepcopy(overrides) if isinstance(overrides, dict) else {},
            "config": config,
        }

    def _expand_seed_specs(self, combo: dict[str, Any]) -> list[dict[str, Any]]:
        """Expand one valid combo across configured seeds."""

        seeds = self._seed_values()
        if not seeds:
            return [self._build_spec(combo)]
        specs: list[dict[str, Any]] = []
        for seed in seeds:
            seeded_combo = dict(combo)
            seeded_combo["seed"] = seed
            specs.append(self._build_spec(seeded_combo))
        return specs

    def _resolve_component(self, group: str, alias: str) -> dict[str, Any]:
        """Resolve one component alias into an experiment-ready component spec."""

        if self._is_paper_schema():
            return self._resolve_paper_component(group, alias)

        definitions = self.config.get(group)
        if isinstance(definitions, dict) and alias in definitions:
            component = dict(definitions[alias])
        else:
            singular_group = group[:-1] if group.endswith("s") else group
            defaults = self.config.get("defaults", {})
            default_component = defaults.get(singular_group)
            if isinstance(default_component, dict) and default_component.get("name", default_component.get("alias")) == alias:
                component = dict(default_component)
            else:
                raise KeyError(f"Unknown ablation component alias `{alias}` in group `{group}`.")

        params = dict(component.get("params", {}))
        root = params.get("root")
        if isinstance(root, str):
            root_path = Path(root)
            if not root_path.is_absolute():
                params["root"] = str((self.source_path.parent / root_path).resolve())
        component["params"] = params
        return component

    def _resolve_paper_component(self, group: str, alias: str) -> dict[str, Any]:
        """Resolve one component alias from the compact paper-style schema."""

        if group == "datasets":
            dataset_entries = self.config.get("datasets", [])
            for entry in dataset_entries:
                if isinstance(entry, dict) and entry.get("name") == alias:
                    dataset_config_path = entry.get("dataset_config")
                    if not isinstance(dataset_config_path, str):
                        raise ValueError(f"Dataset `{alias}` must define `dataset_config`.")
                    return self._load_dataset_component(dataset_config_path)
            raise KeyError(f"Unknown paper-schema dataset alias `{alias}`.")

        if group == "representations":
            return self._paper_catalog()["representations"][alias]
        if group == "normality":
            return self._paper_catalog()["normality"][alias]
        if group == "evidence":
            return self._paper_catalog()["evidence"][alias]
        if group == "protocols":
            return self._paper_catalog()["protocols"][alias]
        if group == "evaluators":
            return self._paper_catalog()["evaluators"][alias]

        raise KeyError(f"Unsupported paper-schema component group `{group}`.")

    def _load_dataset_component(self, config_path: str) -> dict[str, Any]:
        """Load one dataset component definition from a referenced dataset config file."""

        candidate = Path(config_path)
        candidates = [
            (self.source_path.parent / candidate).resolve(),
            (Path(__file__).resolve().parents[3] / candidate).resolve(),
        ]
        for path in candidates:
            if path.exists():
                config = load_yaml_config(path)
                component = {
                    "name": str(config.get("name", "mvtec_single_class")),
                    "params": dict(config.get("params", {})),
                }
                params = component["params"]
                root = params.get("root")
                if isinstance(root, str):
                    root_path = Path(root)
                    if not root_path.is_absolute():
                        repo_root = Path(__file__).resolve().parents[3]
                        repo_relative = (repo_root / root_path).resolve()
                        config_relative = (path.parent / root_path).resolve()
                        params["root"] = str(repo_relative if repo_relative.exists() else config_relative)
                component["params"] = params
                return component
        raise FileNotFoundError(f"Dataset config not found for path `{config_path}`.")

    def _paper_catalog(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Return the built-in component catalog used by the compact paper schema."""

        return {
            "representations": {
                "pixel": {"name": "pixel", "params": {}},
                "feature": {"name": "feature", "params": {"pretrained": False, "freeze": True}},
            },
            "normality": {
                "feature_memory": {"name": "feature_memory", "params": {}},
                "autoencoder": {
                    "name": "autoencoder",
                    "params": {
                        "input_channels": 3,
                        "hidden_channels": 4,
                        "latent_channels": 8,
                        "learning_rate": 0.001,
                        "epochs": 1,
                        "batch_size": 2,
                    },
                },
                "diffusion_basic": {
                    "name": "diffusion_basic",
                    "params": {
                        "input_channels": 3,
                        "hidden_channels": 8,
                        "learning_rate": 0.001,
                        "epochs": 1,
                        "batch_size": 2,
                        "noise_level": 0.2,
                    },
                },
                "diffusion_inversion_basic": {
                    "name": "diffusion_inversion_basic",
                    "params": {
                        "input_channels": 3,
                        "hidden_channels": 8,
                        "learning_rate": 0.001,
                        "epochs": 1,
                        "batch_size": 2,
                        "noise_level": 0.2,
                        "num_steps": 4,
                        "step_size": 0.1,
                    },
                },
                "reference_basic": {
                    "name": "reference_basic",
                    "params": {
                        "input_channels": 3,
                        "hidden_channels": 8,
                        "learning_rate": 0.001,
                        "epochs": 1,
                        "batch_size": 2,
                    },
                },
                "reference_diffusion_basic": {
                    "name": "reference_diffusion_basic",
                    "params": {
                        "input_channels": 3,
                        "hidden_channels": 8,
                        "learning_rate": 0.001,
                        "epochs": 1,
                        "batch_size": 2,
                        "noise_level": 0.2,
                    },
                },
            },
            "evidence": {
                "feature_distance": {"name": "feature_distance", "params": {"aggregator": "max"}},
                "reconstruction_residual": {"name": "reconstruction_residual", "params": {"aggregator": "mean"}},
                "noise_residual": {"name": "noise_residual", "params": {"aggregator": "mean"}},
                "path_cost": {"name": "path_cost", "params": {"aggregator": "mean"}},
                "direction_mismatch": {
                    "name": "direction_mismatch",
                    "params": {"aggregator": "mean", "direction_reduce": "sum"},
                },
                "conditional_violation": {"name": "conditional_violation", "params": {"aggregator": "mean"}},
            },
            "protocols": {"one_class": {"name": "one_class", "params": {}}},
            "evaluators": {
                "default": {"name": "basic_ad", "params": {}},
                "basic_ad": {"name": "basic_ad", "params": {}},
            },
        }

    def _resolve_evaluator_alias(self) -> str:
        """Resolve the evaluator alias used by the compact paper schema."""

        evaluation = self.config.get("evaluation", "default")
        return str(evaluation)

    def _default_alias(self, group_name: str) -> str:
        """Return the default alias configured for a component group."""

        defaults = self.config.get("defaults", {})
        group_defaults = defaults.get(group_name, {})
        alias = group_defaults.get("alias")
        if isinstance(alias, str) and alias:
            return alias

        default_name = group_defaults.get("name")
        if isinstance(default_name, str) and default_name:
            return default_name

        raise ValueError(f"Ablation matrix defaults must define `{group_name}.alias` or `{group_name}.name`.")

    def _seed_values(self) -> list[int]:
        """Return the configured seed list for multi-seed expansion."""

        raw_seeds = self.config.get("seeds", [])
        if raw_seeds in (None, []):
            return []
        if not isinstance(raw_seeds, list) or not all(isinstance(seed, int) for seed in raw_seeds):
            raise ValueError("Ablation matrix `seeds` must be a list of integers.")
        return list(raw_seeds)

    def _normalize_compatibility_rules(self, rules: dict[str, Any]) -> dict[str, Any]:
        """Normalize compact compatibility syntax into the internal rule structure."""

        if not isinstance(rules, dict):
            return {}
        if "normality_evidence" in rules or "representation_normality" in rules or "dataset_protocol" in rules:
            return rules
        return {"normality_evidence": rules}

    def _is_paper_schema(self) -> bool:
        """Return whether the loaded config uses the compact paper-style schema."""

        datasets = self.config.get("datasets")
        return isinstance(datasets, list)

    def _matrix_overrides(self) -> dict[str, Any]:
        """Return normalized matrix-level overrides when configured."""

        overrides = self.config.get("overrides", {})
        return deepcopy(overrides) if isinstance(overrides, dict) else {}

    def _resolve_runtime_config(self, overrides: Any) -> Any:
        """Resolve the effective runtime config value for one expanded spec."""

        if isinstance(overrides, dict) and "runtime_config" in overrides:
            return overrides["runtime_config"]
        return self.config.get("runtime_config")

    def _merge_nested_mappings(self, base: Any, override: Any) -> dict[str, Any]:
        """Recursively merge two mapping-like override payloads."""

        base_mapping = base if isinstance(base, dict) else {}
        override_mapping = override if isinstance(override, dict) else {}
        merged: dict[str, Any] = deepcopy(base_mapping)
        for key, value in override_mapping.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merge_nested_mappings(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged
