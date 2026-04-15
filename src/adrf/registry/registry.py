"""Simple grouped registry used to build framework components from config."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DEFAULT_GROUPS = (
    "dataset",
    "representation",
    "normality",
    "evidence",
    "protocol",
    "evaluator",
)


@dataclass(slots=True)
class Registry:
    """Store named objects under explicit framework groups."""

    groups: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {group: {} for group in DEFAULT_GROUPS}
    )

    def register(self, group: str, name: str, obj: Any) -> None:
        """Register an object under a group/name pair."""

        bucket = self.groups.get(group)
        if bucket is None:
            raise KeyError(f"Registry received unknown group '{group}'.")
        if name in bucket:
            raise KeyError(f"Object '{name}' is already registered in group '{group}'.")
        bucket[name] = obj

    def get(self, group: str, name: str) -> Any:
        """Retrieve a registered object by group and name."""

        if group not in self.groups:
            raise KeyError(f"Registry received unknown group '{group}'.")
        if not self.exists(group, name):
            raise KeyError(f"Object '{name}' is not registered in group '{group}'.")
        return self.groups[group][name]

    def exists(self, group: str, name: str) -> bool:
        """Return whether a group/name pair exists."""

        return name in self.groups.get(group, {})

    def list_available(self, group: str) -> list[str]:
        """List registered names for a group in sorted order."""

        return sorted(self.groups.get(group, {}))
