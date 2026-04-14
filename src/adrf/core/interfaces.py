"""Abstract interfaces for the main research pipeline stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample

if TYPE_CHECKING:
    from adrf.representation.contracts import RepresentationBatch, RepresentationOutput, RepresentationProvenance


class Representation(ABC):
    """Convert a sample into a representation dictionary."""

    @abstractmethod
    def __call__(self, sample: Sample) -> dict[str, Any]:
        """Produce a representation for a single sample."""


class RepresentationModel(ABC):
    space: str
    trainable: bool

    @abstractmethod
    def encode_batch(self, samples: Sequence[Sample]) -> RepresentationBatch:
        """Produce one batched representation payload."""

    def encode_sample(self, sample: Sample) -> RepresentationOutput:
        return self.encode_batch([sample]).unbind()[0]

    @abstractmethod
    def describe(self) -> RepresentationProvenance:
        """Return the static provenance carried by this representation."""


class NormalityModel(ABC):
    fit_mode: Literal["offline", "joint"] = "offline"
    accepted_spaces: frozenset[str] = frozenset()
    accepted_tensor_ranks: frozenset[int] = frozenset()
    requires_detached_representation: bool = True

    @abstractmethod
    def fit(
        self,
        representations: Iterable[RepresentationOutput],
        samples: Iterable[Sample] | None = None,
    ) -> None:
        """Fit the model using normal training representations."""

    def configure_joint_training(self, representation_model: Any) -> None:
        del representation_model
        raise RuntimeError(f"{type(self).__name__} does not implement joint training.")

    def fit_batch(self, representations: RepresentationBatch, samples: Sequence[Sample]) -> dict[str, float]:
        del representations, samples
        raise RuntimeError(f"{type(self).__name__} does not implement fit_batch().")

    @abstractmethod
    def infer(self, sample: Sample, representation: RepresentationOutput) -> NormalityArtifacts:
        """Infer normality artifacts for a represented sample."""


class EvidenceModel(ABC):
    """Convert artifacts into anomaly evidence without using model internals."""

    @abstractmethod
    def predict(self, sample: Sample, artifacts: NormalityArtifacts) -> dict[str, Any]:
        """Produce evidence such as anomaly maps and image-level scores."""


class Protocol(ABC):
    """Coordinate the runner-owned pipeline stages under one runtime contract."""

    @abstractmethod
    def train_epoch(self, runner: Any) -> dict[str, Any]:
        """Execute the training phase for a configured experiment runner."""

    @abstractmethod
    def evaluate(self, runner: Any) -> dict[str, float]:
        """Execute evaluation for a configured experiment runner."""

    @abstractmethod
    def run(self, runner: Any) -> dict[str, Any]:
        """Execute the full protocol lifecycle for a configured experiment runner."""


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
