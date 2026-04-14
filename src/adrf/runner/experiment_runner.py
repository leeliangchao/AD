"""Minimal experiment runner for MVP baseline execution."""

from __future__ import annotations

import random
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from adrf.checkpoint.io import save_model_checkpoint
from adrf.core.sample import Sample
from adrf.data.datamodule import MVTecDataModule
from adrf.evaluation.evaluator import BasicADEvaluator
from adrf.evidence.conditional_violation import ConditionalViolationEvidence
from adrf.evidence.direction_mismatch import DirectionMismatchEvidence
from adrf.evidence.feature_distance import FeatureDistanceEvidence
from adrf.evidence.noise_residual import NoiseResidualEvidence
from adrf.evidence.path_cost import PathCostEvidence
from adrf.evidence.reconstruction_residual import ReconstructionResidualEvidence
from adrf.logging.base import BaseLogger
from adrf.logging.null_logger import NullLogger
from adrf.logging.run_logger import RunLogger
from adrf.normality.autoencoder import AutoEncoderNormality
from adrf.normality.diffusion_basic import DiffusionBasicNormality
from adrf.normality.diffusion_inversion_basic import DiffusionInversionBasicNormality
from adrf.normality.feature_memory import FeatureMemoryNormality
from adrf.normality.reference_basic import ReferenceBasicNormality
from adrf.normality.reference_diffusion_basic import ReferenceDiffusionBasicNormality
from adrf.protocol.one_class import OneClassProtocol
from adrf.registry.registry import Registry
from adrf.reporting.report import export_experiment_report
from adrf.representation.feature import FeatureRepresentation
from adrf.representation.pixel import PixelRepresentation
from adrf.utils.config import instantiate_component, load_yaml_config
from adrf.utils.distributed import (
    DistributedRuntimeContext,
    destroy_distributed_context,
    initialize_distributed_context,
    resolve_distributed_context,
)
from adrf.utils.device import resolve_device
from adrf.utils.runtime import (
    RuntimeRepresentationAdapter,
    configure_trainable_runtime,
    load_runtime_profile,
    resolve_dataloader_runtime,
)


def build_default_registry() -> Registry:
    """Build the registry used by the minimal experiment runner."""

    registry = Registry()
    registry.register("dataset", "mvtec_single_class", MVTecDataModule)
    registry.register("representation", "pixel", PixelRepresentation)
    registry.register("representation", "feature", FeatureRepresentation)
    registry.register("normality", "feature_memory", FeatureMemoryNormality)
    registry.register("normality", "autoencoder", AutoEncoderNormality)
    registry.register("normality", "diffusion_basic", DiffusionBasicNormality)
    registry.register("normality", "diffusion_inversion_basic", DiffusionInversionBasicNormality)
    registry.register("normality", "reference_basic", ReferenceBasicNormality)
    registry.register("normality", "reference_diffusion_basic", ReferenceDiffusionBasicNormality)
    registry.register("evidence", "conditional_violation", ConditionalViolationEvidence)
    registry.register("evidence", "direction_mismatch", DirectionMismatchEvidence)
    registry.register("evidence", "feature_distance", FeatureDistanceEvidence)
    registry.register("evidence", "reconstruction_residual", ReconstructionResidualEvidence)
    registry.register("evidence", "noise_residual", NoiseResidualEvidence)
    registry.register("evidence", "path_cost", PathCostEvidence)
    registry.register("protocol", "one_class", OneClassProtocol)
    registry.register("evaluator", "basic_ad", BasicADEvaluator)
    return registry


class ExperimentRunner:
    """Assemble configured components and execute one experiment run."""

    def __init__(
        self,
        config: str | Path | Mapping[str, Any],
        registry: Registry | None = None,
        logger: BaseLogger | Sequence[BaseLogger] | None = None,
        output_root: str | Path = "outputs/runs",
        run_name: str | None = None,
        seed: int | None = None,
        runtime_config: str | Path | Mapping[str, Any] | None = None,
    ) -> None:
        self.config_path = Path(config).resolve() if isinstance(config, (str, Path)) else None
        self.config = load_yaml_config(config) if isinstance(config, (str, Path)) else dict(config)
        self.registry = registry or build_default_registry()
        self.raw_overrides = self._extract_overrides(self.config)
        self.config = self._apply_budget_overrides(self.config, self.raw_overrides)
        self.runtime_profile = self._resolve_runtime_profile(runtime_config)
        self.runtime_profile = self._apply_runtime_overrides(self.runtime_profile, self.raw_overrides)
        self.distributed_context = resolve_distributed_context(self.runtime_profile)
        self.runtime_profile["distributed"] = self.distributed_context.as_runtime_config()
        self.loggers = self._normalize_loggers(
            logger,
            output_root,
            local_artifacts_enabled=self.distributed_context.is_primary,
        )
        self.logger = self._primary_logger(self.loggers)
        self.run_name = run_name or self._derive_run_name()
        self.seed = seed if seed is not None else self._derive_seed()
        self.device, self.device_info = resolve_device(self.runtime_profile, self.distributed_context)
        self.budget_info = self._collect_budget_info(self.config, self.raw_overrides)
        self.runtime_stats: dict[str, Any] = {}
        self.run_dir: Path | None = None
        self.datamodule: Any | None = None
        self.representation: Any | None = None
        self.normality: Any | None = None
        self.evidence: Any | None = None
        self.evaluator: Any | None = None
        self.protocol: Any | None = None
        self.distributed_training_enabled = False

    def setup(self) -> None:
        """Instantiate all configured components."""

        self.distributed_context = initialize_distributed_context(self.distributed_context, self.device)
        self.runtime_profile["distributed"] = self.distributed_context.as_runtime_config()
        datamodule_spec = self._resolve_datamodule_spec(self.config["datamodule"])
        self.datamodule = instantiate_component(
            datamodule_spec,
            registry=self.registry,
            group="dataset",
        )
        self.datamodule.runtime = self.runtime_profile

        representation = instantiate_component(
            self.config["representation"],
            registry=self.registry,
            group="representation",
        )
        self.representation = RuntimeRepresentationAdapter(
            representation,
            self.device,
            non_blocking=bool(resolve_dataloader_runtime(self.runtime_profile).get("non_blocking", False)),
        )

        self.normality = instantiate_component(
            self.config["normality"],
            registry=self.registry,
            group="normality",
        )
        self._validate_representation_normality_contract()
        configure_trainable_runtime(
            self.normality,
            device=self.device,
            amp_enabled=bool(self.device_info["amp_enabled"]),
            distributed_context=self.distributed_context,
        )
        self.distributed_training_enabled = bool(getattr(self.normality, "distributed_training_enabled", False))

        self.evidence = instantiate_component(
            self.config["evidence"],
            registry=self.registry,
            group="evidence",
        )
        self.evaluator = instantiate_component(
            self.config["evaluator"],
            registry=self.registry,
            group="evaluator",
        )
        self.protocol = instantiate_component(
            self.config["protocol"],
            registry=self.registry,
            group="protocol",
        )

    def _validate_representation_normality_contract(self) -> None:
        """Fail fast when representation outputs cannot satisfy the normality model contract."""

        if self.representation is None or self.normality is None:
            raise RuntimeError("Representation and normality must be instantiated before contract validation.")

        representation_space = getattr(self.representation, "space", None)
        accepted_spaces = getattr(self.normality, "accepted_spaces", frozenset())
        if accepted_spaces and representation_space not in accepted_spaces:
            accepted = ", ".join(f"`{candidate}`" for candidate in sorted(accepted_spaces))
            raise ValueError(
                f"{type(self.normality).__name__} requires representation space {accepted}; "
                f"got `{representation_space}` from {type(getattr(self.representation, 'representation', self.representation)).__name__}."
            )

        probe_output = self._probe_representation_contract_output()
        accepted_tensor_ranks = getattr(self.normality, "accepted_tensor_ranks", frozenset())
        if accepted_tensor_ranks and probe_output.tensor.ndim not in accepted_tensor_ranks:
            accepted = ", ".join(str(rank) for rank in sorted(accepted_tensor_ranks))
            raise ValueError(
                f"{type(self.normality).__name__} requires representation tensor rank in {{{accepted}}}; "
                f"got {probe_output.tensor.ndim} from {type(getattr(self.representation, 'representation', self.representation)).__name__}."
            )

        fit_mode = str(getattr(self.normality, "fit_mode", "offline"))
        if (
            fit_mode == "offline"
            and bool(getattr(self.normality, "requires_detached_representation", False))
            and probe_output.requires_grad
        ):
            raise ValueError(
                f"{type(self.normality).__name__} requires detached representations for offline fit mode, "
                f"but {type(getattr(self.representation, 'representation', self.representation)).__name__} emits a trainable representation."
            )

    def _probe_representation_contract_output(self) -> Any:
        """Inspect one representation output without mutating representation training state."""

        if self.representation is None:
            raise RuntimeError("Representation must be instantiated before probing its contract.")

        representation_model = getattr(self.representation, "representation", self.representation)
        if not isinstance(representation_model, nn.Module):
            return self.representation.encode_batch(self._make_contract_probe_batch()).unbind()[0]

        training_states = {module: module.training for module in representation_model.modules()}
        try:
            representation_model.eval()
            return self.representation.encode_batch(self._make_contract_probe_batch()).unbind()[0]
        finally:
            for module, was_training in training_states.items():
                module.train(was_training)

    def _make_contract_probe_batch(self) -> list[Sample]:
        """Create lightweight samples for representation/normality compatibility checks."""

        image_size = getattr(self.representation, "input_image_size", None)
        if not isinstance(image_size, tuple) or len(image_size) != 2:
            image_size_cfg = self.config.get("datamodule", {}).get("params", {}).get("image_size", (32, 32))
            image_size = tuple(int(size) for size in image_size_cfg)
        height, width = int(image_size[0]), int(image_size[1])
        return [
            Sample(image=torch.zeros(3, height, width), sample_id="__adrf_contract_probe__0"),
            Sample(image=torch.zeros(3, height, width), sample_id="__adrf_contract_probe__1"),
        ]

    def train(self) -> dict[str, Any]:
        """Run the training stage through the configured protocol."""

        self._ensure_setup()
        return self.protocol.train_epoch(self)

    def evaluate(self) -> dict[str, float]:
        """Run the evaluation stage through the configured protocol."""

        self._ensure_setup()
        return self.protocol.evaluate(self)

    def run(self) -> dict[str, Any]:
        """Run the full experiment lifecycle."""

        try:
            self._ensure_setup()
            self._initialize_seed()
            self._start_logged_run()
            profiling_cfg = self.runtime_profile.get("profiling", {})
            record_timing = bool(profiling_cfg.get("record_timing", True))
            record_memory = bool(profiling_cfg.get("record_memory", False))
            total_start = time.perf_counter()
            if record_memory and self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            protocol_start = time.perf_counter()
            protocol_results = self.protocol.run(self)
            total_protocol_time = time.perf_counter() - protocol_start
            train_results, evaluation_results = self._normalize_protocol_results(protocol_results)

            train_time = self._maybe_float(train_results.get("train_time"))
            eval_time = self._maybe_float(train_results.get("eval_time"))
            if train_time is None and eval_time is not None:
                train_time = max(total_protocol_time - eval_time, 0.0)
            if eval_time is None and train_time is not None:
                eval_time = max(total_protocol_time - train_time, 0.0)
            if train_time is None and eval_time is None:
                train_time = total_protocol_time
                eval_time = 0.0
            total_time = time.perf_counter() - total_start
            peak_memory = (
                int(torch.cuda.max_memory_allocated(self.device))
                if record_memory and self.device.type == "cuda"
                else None
            )
            self.runtime_stats = {
                "profile_name": self.runtime_profile.get("name", "default"),
                **self.device_info,
                "distributed_enabled": self.distributed_context.enabled,
                "rank": self.distributed_context.rank,
                "local_rank": self.distributed_context.local_rank,
                "world_size": self.distributed_context.world_size,
                "dataloader": resolve_dataloader_runtime(self.runtime_profile),
                "train_time_s": train_time if record_timing else None,
                "eval_time_s": eval_time if record_timing else None,
                "total_time_s": total_time if record_timing else None,
                "peak_memory_bytes": peak_memory,
            }

            results = {"train": train_results, "evaluation": evaluation_results}
            self._log_metrics(results)
            self._log_run_info({"runtime": self.runtime_stats, "budget": self.budget_info})
            self._save_checkpoint_if_supported()
            self._finish_loggers(status="completed")
            if self.run_dir is not None:
                export_experiment_report(self.run_dir)
            return results
        except Exception:
            self._finish_loggers(status="failed")
            raise
        finally:
            destroy_distributed_context(self.distributed_context)

    def _ensure_setup(self) -> None:
        """Instantiate components lazily when needed."""

        if self.datamodule is None:
            self.setup()

    @staticmethod
    def _normalize_protocol_results(results: Any) -> tuple[dict[str, Any], dict[str, float]]:
        """Validate and normalize the protocol return payload."""

        if not isinstance(results, Mapping):
            raise TypeError("Protocol.run() must return a mapping with `train` and `evaluation` keys.")

        train_results = results.get("train")
        evaluation_results = results.get("evaluation")
        if not isinstance(train_results, Mapping):
            raise TypeError("Protocol.run() must return a mapping under the `train` key.")
        if not isinstance(evaluation_results, Mapping):
            raise TypeError("Protocol.run() must return a mapping under the `evaluation` key.")

        return dict(train_results), {
            str(key): float(value)
            for key, value in evaluation_results.items()
            if isinstance(value, (int, float))
        }

    @staticmethod
    def _maybe_float(value: Any) -> float | None:
        """Return a float when the value is numeric."""

        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _resolve_datamodule_spec(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        """Resolve datamodule filesystem paths relative to the config location."""

        resolved_spec = dict(spec)
        params = dict(spec.get("params", {}))
        root = params.get("root")
        if isinstance(root, str) and self.config_path is not None:
            root_path = Path(root)
            if not root_path.is_absolute():
                repo_root = Path(__file__).resolve().parents[3]
                repo_relative = (repo_root / root_path).resolve()
                config_relative = (self.config_path.parent / root_path).resolve()
                params["root"] = str(repo_relative if repo_relative.exists() else config_relative)
        resolved_spec["params"] = params
        return resolved_spec

    def _start_logged_run(self) -> None:
        """Create the local run directory and persist initial metadata."""

        if not self.distributed_context.is_primary:
            return
        if self.run_dir is not None:
            return

        run_info = {
            "config_path": str(self.config_path) if self.config_path is not None else None,
            "seed": self.seed,
            "runtime": {
                "profile_name": self.runtime_profile.get("name", "default"),
                **self.device_info,
                "distributed": self.distributed_context.as_runtime_config(),
                "dataloader": resolve_dataloader_runtime(self.runtime_profile),
            },
            "budget": self.budget_info,
            "components": {
                key: self.config[key]["name"]
                for key in ("datamodule", "representation", "normality", "evidence", "protocol", "evaluator")
                if key in self.config and isinstance(self.config[key], Mapping) and "name" in self.config[key]
            },
        }
        for logger in self.loggers:
            logger.start_run(self.run_name, self.config, run_info=run_info)
        self.run_dir = self.logger.resolve_artifact_path("run_info.json").parent

    def _save_checkpoint_if_supported(self) -> None:
        """Persist a minimal checkpoint for trainable normality models."""

        if self.run_dir is None or self.logger is None or not self.distributed_context.is_primary:
            return

        checkpoint_path = self.logger.resolve_artifact_path("checkpoints/normality.pt")
        if save_model_checkpoint(self.normality, checkpoint_path):
            for logger in self.loggers:
                logger.log_artifact(checkpoint_path, artifact_type="checkpoint")

    def _derive_run_name(self) -> str:
        """Derive a stable run name from config metadata."""

        if self.config_path is not None:
            return self.config_path.stem
        return str(self.config.get("name", "experiment"))

    def _derive_seed(self) -> int | None:
        """Resolve the configured random seed when present."""

        seed = self.config.get("seed")
        return int(seed) if isinstance(seed, int) else None

    def _resolve_runtime_profile(
        self,
        runtime_config: str | Path | Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Resolve the runtime profile from explicit input or config metadata."""

        candidate = runtime_config if runtime_config is not None else self.config.get("runtime_config")
        if candidate is None:
            return load_runtime_profile(None)
        if isinstance(candidate, Mapping):
            return load_runtime_profile(dict(candidate))

        runtime_path = Path(candidate)
        if not runtime_path.is_absolute():
            repo_root = Path(__file__).resolve().parents[3]
            repo_relative = (repo_root / runtime_path).resolve()
            if repo_relative.exists():
                runtime_path = repo_relative
            elif self.config_path is not None:
                runtime_path = (self.config_path.parent / runtime_path).resolve()
        return load_runtime_profile(runtime_path)

    def _initialize_seed(self) -> None:
        """Seed Python, NumPy, and PyTorch when a seed is configured."""

        if self.seed is None:
            return
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def _log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Fan out metric logging to all configured loggers."""

        for logger in self.loggers:
            logger.log_metrics(metrics, step=step)

    def _log_run_info(self, updates: dict[str, Any]) -> None:
        """Fan out run metadata updates to all configured loggers."""

        for logger in self.loggers:
            logger.log_run_info(updates)

    def _finish_loggers(self, status: str) -> None:
        """Fan out run finalization to all configured loggers."""

        for logger in self.loggers:
            logger.finish_run(status=status)

    @staticmethod
    def _normalize_loggers(
        logger: BaseLogger | Sequence[BaseLogger] | None,
        output_root: str | Path,
        *,
        local_artifacts_enabled: bool,
    ) -> list[BaseLogger]:
        """Normalize logger input into a list while ensuring a local RunLogger exists."""

        if not local_artifacts_enabled:
            return [NullLogger()]

        if logger is None:
            return [RunLogger(base_dir=output_root)]

        if isinstance(logger, BaseLogger):
            loggers = [logger]
        else:
            loggers = list(logger)
            if not loggers:
                loggers = []

        if not any(isinstance(item, RunLogger) for item in loggers):
            loggers.insert(0, RunLogger(base_dir=output_root))
        return loggers

    @staticmethod
    def _primary_logger(loggers: Sequence[BaseLogger]) -> RunLogger | None:
        """Return the local RunLogger that owns filesystem artifacts."""

        for logger in loggers:
            if isinstance(logger, RunLogger):
                return logger
        return None

    @staticmethod
    def _extract_overrides(config: Mapping[str, Any]) -> dict[str, Any]:
        """Extract a top-level override payload from one experiment config."""

        overrides = config.get("overrides", {})
        return deepcopy(overrides) if isinstance(overrides, Mapping) else {}

    def _apply_budget_overrides(
        self,
        config: Mapping[str, Any],
        overrides: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Apply matrix-level budget overrides onto the runnable experiment config."""

        resolved = deepcopy(dict(config))
        resolved.pop("overrides", None)
        if not overrides:
            return resolved

        runtime_override = overrides.get("runtime_config")
        if runtime_override is not None:
            resolved["runtime_config"] = deepcopy(runtime_override)

        dataset_overrides = overrides.get("dataset")
        if isinstance(dataset_overrides, Mapping):
            for key, value in dataset_overrides.items():
                self._set_component_param(resolved, "datamodule", str(key), value)

        dataloader_overrides = overrides.get("dataloader")
        if isinstance(dataloader_overrides, Mapping):
            for key, value in dataloader_overrides.items():
                self._set_component_param(resolved, "datamodule", str(key), value)
            if "batch_size" in dataloader_overrides:
                self._set_component_param(
                    resolved,
                    "normality",
                    "batch_size",
                    dataloader_overrides["batch_size"],
                    only_if_supported=True,
                )

        trainer_overrides = overrides.get("trainer")
        if isinstance(trainer_overrides, Mapping):
            if "max_epochs" in trainer_overrides:
                self._set_component_param(
                    resolved,
                    "normality",
                    "epochs",
                    trainer_overrides["max_epochs"],
                    only_if_supported=True,
                )
            elif "epochs" in trainer_overrides:
                self._set_component_param(
                    resolved,
                    "normality",
                    "epochs",
                    trainer_overrides["epochs"],
                    only_if_supported=True,
                )

        optimization_overrides = overrides.get("optimization")
        if isinstance(optimization_overrides, Mapping):
            for key, value in optimization_overrides.items():
                param_name = "learning_rate" if key == "lr" else str(key)
                self._set_component_param(
                    resolved,
                    "normality",
                    param_name,
                    value,
                    only_if_supported=True,
                )

        for key in ("normality", "diffusion"):
            section = overrides.get(key)
            if isinstance(section, Mapping):
                for param_key, value in section.items():
                    self._set_component_param(
                        resolved,
                        "normality",
                        str(param_key),
                        value,
                        only_if_supported=True,
                    )

        for component_key in ("datamodule", "representation", "normality", "evidence", "protocol", "evaluator"):
            section = overrides.get(component_key)
            if isinstance(section, Mapping):
                self._merge_component_override(resolved, component_key, section)

        return resolved

    def _collect_budget_info(
        self,
        config: Mapping[str, Any],
        overrides: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Summarize the effective training budget after overrides are applied."""

        datamodule_params = self._component_params(config, "datamodule")
        normality_params = self._component_params(config, "normality")
        dataloader_runtime = resolve_dataloader_runtime(
            self.runtime_profile,
            defaults={"batch_size": datamodule_params.get("batch_size")},
        )
        budget: dict[str, Any] = {
            "max_epochs": normality_params.get("epochs"),
            "batch_size": dataloader_runtime.get("batch_size", datamodule_params.get("batch_size", normality_params.get("batch_size"))),
            "lr": normality_params.get("learning_rate"),
            "runtime_config": config.get("runtime_config"),
            "num_steps": normality_params.get("num_steps"),
            "backend": normality_params.get("backend"),
            "image_size": datamodule_params.get("image_size"),
        }
        if overrides:
            budget["overrides"] = deepcopy(dict(overrides))
        return {key: value for key, value in budget.items() if value is not None}

    @staticmethod
    def _merge_component_override(
        config: dict[str, Any],
        component_key: str,
        override: Mapping[str, Any],
    ) -> None:
        """Merge one component-level override mapping into the config."""

        existing = config.get(component_key)
        if not isinstance(existing, Mapping):
            config[component_key] = deepcopy(dict(override))
            return

        merged = deepcopy(dict(existing))
        override_params = override.get("params", {})
        if isinstance(override_params, Mapping):
            params = dict(merged.get("params", {}))
            params.update(deepcopy(dict(override_params)))
            merged["params"] = params
        for key, value in override.items():
            if key != "params":
                merged[str(key)] = deepcopy(value)
        config[component_key] = merged

    def _set_component_param(
        self,
        config: dict[str, Any],
        component_key: str,
        param_key: str,
        value: Any,
        only_if_supported: bool = False,
    ) -> None:
        """Set one nested component param on the config."""

        component = config.get(component_key)
        if not isinstance(component, Mapping):
            return
        if only_if_supported and not self._component_supports_param(config, component_key, param_key):
            return
        merged = deepcopy(dict(component))
        params = dict(merged.get("params", {}))
        params[param_key] = deepcopy(value)
        merged["params"] = params
        config[component_key] = merged

    @staticmethod
    def _component_params(config: Mapping[str, Any], component_key: str) -> dict[str, Any]:
        """Return the params mapping for one component when available."""

        component = config.get(component_key, {})
        if not isinstance(component, Mapping):
            return {}
        params = component.get("params", {})
        return dict(params) if isinstance(params, Mapping) else {}

    @staticmethod
    def _apply_runtime_overrides(
        runtime_profile: Mapping[str, Any],
        overrides: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Apply runtime-relevant budget overrides onto the resolved runtime profile."""

        resolved_runtime = deepcopy(dict(runtime_profile))
        dataloader_overrides = overrides.get("dataloader")
        if isinstance(dataloader_overrides, Mapping):
            dataloader = dict(resolved_runtime.get("dataloader", {}))
            for key, value in dataloader_overrides.items():
                dataloader[str(key)] = deepcopy(value)
            resolved_runtime["dataloader"] = dataloader
        return resolved_runtime

    def _component_supports_param(
        self,
        config: Mapping[str, Any],
        component_key: str,
        param_key: str,
    ) -> bool:
        """Return whether a configured component can accept one constructor param."""

        component = config.get(component_key, {})
        if not isinstance(component, Mapping):
            return False

        params = component.get("params", {})
        if isinstance(params, Mapping) and param_key in params:
            return True

        component_name = component.get("name")
        if not isinstance(component_name, str) or not component_name:
            return False

        group_map = {
            "datamodule": "dataset",
            "representation": "representation",
            "normality": "normality",
            "evidence": "evidence",
            "protocol": "protocol",
            "evaluator": "evaluator",
        }
        group = group_map.get(component_key)
        if group is None:
            return False

        try:
            target = self.registry.get(group, component_name)
        except KeyError:
            return False
        if not callable(target):
            return False

        signature = inspect.signature(target)
        if param_key in signature.parameters:
            return True
        return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
