"""Exchange object emitted by normality models and consumed by evidence models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from adrf.core.typing import CapabilitySet, DataDict


@dataclass(slots=True)
class NormalityArtifacts:
    """Structured carrier for intermediate outputs shared between pipeline stages."""

    context: DataDict = field(default_factory=dict)
    representation: DataDict = field(default_factory=dict)
    primary: DataDict = field(default_factory=dict)
    auxiliary: DataDict = field(default_factory=dict)
    diagnostics: DataDict = field(default_factory=dict)
    capabilities: CapabilitySet = field(default_factory=set)

    def has(self, capability: str) -> bool:
        """Return whether a named capability is present."""

        return capability in self.capabilities

    def require(self, *capabilities: str) -> None:
        """Validate that all requested capabilities are available."""

        missing = [capability for capability in capabilities if capability not in self.capabilities]
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise KeyError(f"Missing required capabilities: {missing_text}")

    def get_primary(self, key: str, default: Any = None) -> Any:
        """Read a value from the primary artifact payload."""

        return self.primary.get(key, default)

    def get_aux(self, key: str, default: Any = None) -> Any:
        """Read a value from the auxiliary artifact payload."""

        return self.auxiliary.get(key, default)

    def get_diag(self, key: str, default: Any = None) -> Any:
        """Read a value from the diagnostics artifact payload."""

        return self.diagnostics.get(key, default)

