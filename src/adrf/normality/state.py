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
