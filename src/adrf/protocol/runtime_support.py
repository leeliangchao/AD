from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import Any

from torch import nn

from adrf.protocol.results import TrainSummary
from adrf.utils.distributed import all_gather_objects


@contextmanager
def temporarily_switch_modules_to_eval(*objects: object):
    roots: list[nn.Module] = []
    for obj in objects:
        if isinstance(obj, nn.Module) and all(existing is not obj for existing in roots):
            roots.append(obj)

    if not roots:
        yield
        return

    training_states = {
        module: module.training
        for root in roots
        for module in root.modules()
    }
    try:
        for root in roots:
            root.eval()
        yield
    finally:
        for module, was_training in training_states.items():
            module.train(was_training)


def merge_distributed_train_summary(summary: TrainSummary, distributed_context: Any) -> TrainSummary:
    if (
        distributed_context is None
        or not getattr(distributed_context, "enabled", False)
        or getattr(distributed_context, "world_size", 1) <= 1
    ):
        return summary

    gathered_payloads = all_gather_objects(summary, distributed_context)
    merged = TrainSummary()
    metric_totals: dict[str, float] = defaultdict(float)
    metric_counts: dict[str, float] = defaultdict(float)

    for payload in gathered_payloads:
        if not isinstance(payload, TrainSummary):
            raise TypeError(
                f"merge_distributed_train_summary expected gathered payloads to be TrainSummary objects; "
                f"got {type(payload).__name__}"
            )
        item = payload
        _validate_explicit_metric_weights(item)
        merged.num_train_batches += item.num_train_batches
        merged.num_train_samples += item.num_train_samples
        for key, value in item.metrics.items():
            weight = float(item.metric_weights[key])
            metric_totals[key] += float(value) * weight
            metric_counts[key] += weight

    merged.metrics = {
        key: total / max(metric_counts[key], 1)
        for key, total in metric_totals.items()
    }
    merged.metric_weights = dict(metric_counts)
    return merged


def _validate_explicit_metric_weights(summary: TrainSummary) -> None:
    metric_keys = set(summary.metrics)
    weight_keys = set(summary.metric_weights)
    if metric_keys != weight_keys:
        missing = sorted(metric_keys - weight_keys)
        extra = sorted(weight_keys - metric_keys)
        details: list[str] = []
        if missing:
            details.append(f"missing weights for {', '.join(missing)}")
        if extra:
            details.append(f"unexpected weights for {', '.join(extra)}")
        raise ValueError(f"TrainSummary requires explicit metric weights: {'; '.join(details)}")


def merge_distributed_evaluator_state(evaluator: Any, distributed_context: Any) -> None:
    if (
        distributed_context is None
        or not getattr(distributed_context, "enabled", False)
        or getattr(distributed_context, "world_size", 1) <= 1
    ):
        return

    gathered_states = all_gather_objects(evaluator.state_dict(), distributed_context)
    merged_state = evaluator.merge_states(gathered_states)
    evaluator.load_state_dict(merged_state)
