"""Base helpers for normality models."""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from typing import Any

import torch

from adrf.core.interfaces import NormalityModel


class BaseNormalityModel(NormalityModel, ABC):
    """Common tensor extraction helpers for normality models."""

    @staticmethod
    def require_representation_tensor(representation: Mapping[str, Any]) -> torch.Tensor:
        """Return the representation tensor or raise for invalid payloads."""

        tensor = representation.get("representation")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Normality models expect representation['representation'] to be a torch.Tensor.")
        return tensor

