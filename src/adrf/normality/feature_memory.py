"""Feature memory normality model."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from adrf.core.artifacts import NormalityArtifacts
from adrf.core.sample import Sample
from adrf.normality.base import BaseNormalityModel
from adrf.representation.contracts import RepresentationOutput


class FeatureMemoryNormality(BaseNormalityModel):
    """Store normal feature vectors and measure nearest-memory distance at inference."""

    accepted_spaces = frozenset({"feature"})
    accepted_tensor_ranks = frozenset({1, 3})
    requires_detached_representation = True

    def __init__(self, distance_chunk_size: int | None = 4096) -> None:
        if distance_chunk_size is not None and distance_chunk_size < 1:
            raise ValueError("distance_chunk_size must be positive when provided.")
        self.memory_bank: torch.Tensor | None = None
        self.distance_chunk_size = distance_chunk_size

    def fit(
        self,
        representations: Iterable[RepresentationOutput],
        samples: Iterable[Sample] | None = None,
    ) -> None:
        """Build a memory bank from normal feature representations."""

        del samples
        flattened_batches = [
            self._flatten_feature_tensor(self.require_representation_tensor(representation))[0]
            for representation in representations
        ]
        if not flattened_batches:
            raise ValueError("FeatureMemoryNormality.fit requires at least one representation.")
        self.memory_bank = torch.cat(flattened_batches, dim=0)

    def infer(self, sample: Sample, representation: RepresentationOutput) -> NormalityArtifacts:
        """Infer distances to the learned memory bank and package them as artifacts."""

        if self.memory_bank is None:
            raise RuntimeError("FeatureMemoryNormality must be fitted before infer().")

        feature_response = self.require_representation_tensor(representation)
        flattened, spatial_shape = self._flatten_feature_tensor(feature_response)
        min_distances = self._min_memory_distances(flattened, self.memory_bank)
        if spatial_shape is not None:
            memory_distance = min_distances.reshape(spatial_shape)
        else:
            memory_distance = min_distances

        return NormalityArtifacts(
            context={"sample_id": sample.sample_id, "category": sample.category},
            representation=self.serialize_representation(representation),
            primary={"normality_embedding": flattened.mean(dim=0)},
            auxiliary={
                "feature_response": feature_response,
                "memory_distance": memory_distance,
            },
            capabilities={
                "normality_embedding",
                "feature_response",
                "memory_distance",
            },
        )

    @staticmethod
    def _flatten_feature_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int] | None]:
        """Flatten a feature tensor into [N, D] while tracking spatial layout."""

        if tensor.ndim == 1:
            return tensor.unsqueeze(0), None
        if tensor.ndim == 2:
            return tensor, None
        if tensor.ndim == 3:
            channels, height, width = tensor.shape
            flattened = tensor.permute(1, 2, 0).reshape(height * width, channels)
            return flattened, (height, width)
        raise TypeError("FeatureMemoryNormality expects feature tensors with 1, 2, or 3 dimensions.")

    def _min_memory_distances(self, flattened: torch.Tensor, memory_bank: torch.Tensor) -> torch.Tensor:
        """Compute nearest-memory distances, optionally chunking queries to cap peak memory."""

        if self.distance_chunk_size is None or flattened.shape[0] <= self.distance_chunk_size:
            return torch.cdist(flattened, memory_bank).min(dim=1).values

        chunks = []
        for start in range(0, flattened.shape[0], self.distance_chunk_size):
            query_chunk = flattened[start : start + self.distance_chunk_size]
            chunks.append(torch.cdist(query_chunk, memory_bank).min(dim=1).values)
        return torch.cat(chunks, dim=0)
