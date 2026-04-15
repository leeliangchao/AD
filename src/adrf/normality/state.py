from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from adrf.utils.distributed import DistributedRuntimeContext


@dataclass(slots=True)
class NormalityRuntimeState:
    device: torch.device
    amp_enabled: bool
    grad_scaler: torch.amp.GradScaler
    distributed_context: DistributedRuntimeContext
    distributed_training_enabled: bool


def make_default_normality_runtime_state() -> NormalityRuntimeState:
    from adrf.utils.distributed import DistributedRuntimeContext

    return NormalityRuntimeState(
        device=torch.device("cpu"),
        amp_enabled=False,
        grad_scaler=torch.amp.GradScaler("cuda", enabled=False),
        distributed_context=DistributedRuntimeContext(),
        distributed_training_enabled=False,
    )


def install_normality_runtime_state(model: object, runtime_state: NormalityRuntimeState) -> NormalityRuntimeState:
    setattr(model, "runtime", runtime_state)
    setattr(model, "runtime_device", runtime_state.device)
    setattr(model, "amp_enabled", runtime_state.amp_enabled)
    setattr(model, "grad_scaler", runtime_state.grad_scaler)
    setattr(model, "distributed_context", runtime_state.distributed_context)
    setattr(model, "distributed_training_enabled", runtime_state.distributed_training_enabled)
    return runtime_state
