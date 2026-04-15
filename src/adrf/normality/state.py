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
