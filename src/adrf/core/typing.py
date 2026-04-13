"""Shared type aliases for framework-facing data contracts."""

from __future__ import annotations

from typing import Any, TypeAlias

DataDict: TypeAlias = dict[str, Any]
MetadataDict: TypeAlias = dict[str, Any]
ViewsDict: TypeAlias = dict[str, Any]
CapabilitySet: TypeAlias = set[str]

