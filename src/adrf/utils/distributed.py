"""Distributed runtime helpers for torchrun/DDP execution."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import torch
import torch.distributed as dist


@dataclass(slots=True)
class DistributedRuntimeContext:
    """Normalized distributed runtime metadata."""

    enabled: bool = False
    backend: str = "nccl"
    find_unused_parameters: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    initialized: bool = False

    @property
    def is_primary(self) -> bool:
        """Return whether this process owns primary artifact/logging duties."""

        return self.rank == 0

    def as_runtime_config(self) -> dict[str, Any]:
        """Render the context into runtime-profile-compatible metadata."""

        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "find_unused_parameters": self.find_unused_parameters,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "initialized": self.initialized,
        }


def resolve_distributed_context(runtime_cfg: dict[str, Any]) -> DistributedRuntimeContext:
    """Resolve distributed execution metadata from runtime config and torchrun env."""

    distributed_cfg = runtime_cfg.get("distributed", {})
    if not isinstance(distributed_cfg, dict):
        distributed_cfg = {}

    device_cfg = runtime_cfg.get("device", {})
    if not isinstance(device_cfg, dict):
        device_cfg = {}
    device_type = str(device_cfg.get("type", "auto"))
    device_ids = device_cfg.get("ids")
    normalized_device_ids = list(device_ids) if isinstance(device_ids, list) else []

    env_world_size_raw = os.environ.get("WORLD_SIZE")
    env_enabled = env_world_size_raw is not None and int(env_world_size_raw) > 1
    auto_enabled = device_type == "cuda" and len(normalized_device_ids) > 1
    if not (env_enabled or auto_enabled):
        return DistributedRuntimeContext()

    rank = _resolve_int("RANK", distributed_cfg.get("rank", 0))
    local_rank = _resolve_int("LOCAL_RANK", distributed_cfg.get("local_rank", rank))
    fallback_world_size = distributed_cfg.get("world_size", len(normalized_device_ids) if auto_enabled else 1)
    world_size = _resolve_int("WORLD_SIZE", fallback_world_size)
    if world_size < 1:
        raise ValueError("distributed.world_size must be at least 1.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"distributed rank {rank} is invalid for world_size={world_size}.")

    return DistributedRuntimeContext(
        enabled=True,
        backend=str(distributed_cfg.get("backend", "nccl")),
        find_unused_parameters=bool(distributed_cfg.get("find_unused_parameters", False)),
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        initialized=bool(dist.is_available() and dist.is_initialized()),
    )


def initialize_distributed_context(
    context: DistributedRuntimeContext,
    device: torch.device,
) -> DistributedRuntimeContext:
    """Initialize torch.distributed when the runtime requests it."""

    if not context.enabled or context.world_size <= 1:
        return context
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this environment.")
    if dist.is_initialized():
        context.initialized = True
        return context
    if os.environ.get("WORLD_SIZE") is None:
        raise RuntimeError(
            "Multi-process distributed runtime requires torchrun or the CLI launcher to set "
            "RANK/LOCAL_RANK/WORLD_SIZE."
        )
    if context.backend == "nccl" and device.type != "cuda":
        raise ValueError("distributed backend `nccl` requires a CUDA device.")
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend=context.backend,
        rank=context.rank,
        world_size=context.world_size,
    )
    context.initialized = True
    return context


def destroy_distributed_context(context: DistributedRuntimeContext) -> None:
    """Tear down torch.distributed when this runner initialized it."""

    if not context.enabled or context.world_size <= 1:
        return
    if not dist.is_available():
        return
    if dist.is_initialized():
        dist.destroy_process_group()
    context.initialized = False


def all_gather_objects(payload: Any, context: DistributedRuntimeContext) -> list[Any]:
    """Gather a picklable payload from every rank."""

    if not context.enabled or context.world_size <= 1:
        return [payload]
    gathered: list[Any] = [None] * context.world_size
    dist.all_gather_object(gathered, payload)
    return gathered


def _resolve_int(env_name: str, fallback: Any) -> int:
    """Resolve one integer from the environment or a fallback value."""

    raw = os.environ.get(env_name)
    candidate = raw if raw is not None else fallback
    try:
        return int(candidate)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unable to resolve integer value for {env_name}: {candidate!r}") from exc
