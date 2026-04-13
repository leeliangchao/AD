"""Abstract interfaces for the main research pipeline stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample


class Representation(ABC):
    """Convert a sample into a representation dictionary."""

    @abstractmethod
    def __call__(self, sample: Sample) -> dict[str, Any]:
        """Produce a representation for a single sample."""


class NormalityModel(ABC):
    """Learn normality from representations and emit standardized artifacts."""

    @abstractmethod
    def fit(
        self,
        representations: Iterable[Mapping[str, Any]],
        samples: Iterable[Sample] | None = None,
    ) -> None:
        """Fit the model using normal training representations."""

    @abstractmethod
    def infer(self, sample: Sample, representation: Mapping[str, Any]) -> NormalityArtifacts:
        """Infer normality artifacts for a represented sample."""


class EvidenceModel(ABC):
    """Convert artifacts into anomaly evidence without using model internals."""

    @abstractmethod
    def predict(self, sample: Sample, artifacts: NormalityArtifacts) -> dict[str, Any]:
        """Produce evidence such as anomaly maps and image-level scores."""


class Protocol(ABC):
    """Coordinate data flow across the shared pipeline stages."""

    @abstractmethod
    def run(
        self,
        train_samples: Iterable[Sample],
        test_samples: Iterable[Sample],
        representation: Representation,
        normality: NormalityModel,
        evidence: EvidenceModel,
        evaluator: "Evaluator",
    ) -> dict[str, Any]:
        """Execute training and evaluation for a protocol instance."""


class Evaluator(ABC):
    """Accumulate predictions and report evaluation metrics."""

    @abstractmethod
    def update(self, prediction: Mapping[str, Any], sample: Sample) -> None:
        """Update internal state with one prediction/sample pair."""

    @abstractmethod
    def compute(self) -> dict[str, Any]:
        """Compute final evaluation metrics."""

    @abstractmethod
    def reset(self) -> None:
        """Clear any stored evaluation state."""

