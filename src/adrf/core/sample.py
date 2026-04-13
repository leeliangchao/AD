"""Unified sample schema used across datasets and pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from adrf.core.typing import MetadataDict, ViewsDict


@dataclass(slots=True)
class Sample:
    """Container for a single anomaly-detection sample and its metadata."""

    image: Any
    label: int | None = None
    mask: Any | None = None
    category: str | None = None
    sample_id: str | None = None
    reference: Any | None = None
    views: ViewsDict | None = None
    metadata: MetadataDict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the sample content as a plain dictionary."""

        return {
            "image": self.image,
            "label": self.label,
            "mask": self.mask,
            "category": self.category,
            "sample_id": self.sample_id,
            "reference": self.reference,
            "views": dict(self.views) if self.views is not None else None,
            "metadata": dict(self.metadata),
        }

    def has_reference(self) -> bool:
        """Return whether the sample includes a reference input."""

        return self.reference is not None

    def has_views(self) -> bool:
        """Return whether the sample contains any additional views."""

        return bool(self.views)

