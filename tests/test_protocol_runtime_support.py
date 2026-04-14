import pytest
from types import SimpleNamespace

import torch
from torch import nn

from adrf.protocol.results import TrainSummary
from adrf.protocol.runtime_support import (
    merge_distributed_evaluator_state,
    merge_distributed_train_summary,
    temporarily_switch_modules_to_eval,
)


class _Root(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.batch_norm(inputs)


class _Evaluator:
    def __init__(self) -> None:
        self.loaded_state = None

    def state_dict(self):
        return {"updates": 1}

    def merge_states(self, states):
        return {"updates": sum(int(state["updates"]) for state in states)}

    def load_state_dict(self, state):
        self.loaded_state = state


def test_temporarily_switch_modules_to_eval_restores_training_state() -> None:
    module = _Root()
    before_batches = int(module.batch_norm.num_batches_tracked.item())

    with temporarily_switch_modules_to_eval(module):
        assert module.training is False
        assert module.batch_norm.training is False
        module(torch.ones(1, 1, 4, 4))

    assert module.training is True
    assert module.batch_norm.training is True
    assert int(module.batch_norm.num_batches_tracked.item()) == before_batches


def test_merge_distributed_train_summary_aggregates_counts_without_batch_weighting(monkeypatch) -> None:
    context = SimpleNamespace(enabled=True, world_size=2)
    summary = TrainSummary(
        num_train_batches=1,
        num_train_samples=2,
        metrics={"loss": 1.0},
    )

    monkeypatch.setattr(
        "adrf.protocol.runtime_support.all_gather_objects",
        lambda payload, distributed_context: [
            payload,
            {"num_train_batches": 2, "num_train_samples": 3, "loss": 4.0},
        ],
    )

    assert merge_distributed_train_summary(summary, context) == TrainSummary(
        num_train_batches=3,
        num_train_samples=5,
        metrics={"loss": 2.5},
    )


def test_merge_distributed_train_summary_raises_on_malformed_payload(monkeypatch) -> None:
    context = SimpleNamespace(enabled=True, world_size=2)
    summary = TrainSummary(
        num_train_batches=1,
        num_train_samples=2,
        metrics={"loss": 1.0},
    )

    monkeypatch.setattr(
        "adrf.protocol.runtime_support.all_gather_objects",
        lambda payload, distributed_context: [payload, ["not", "a", "mapping"]],
    )

    with pytest.raises(TypeError, match="gathered payload"):
        merge_distributed_train_summary(summary, context)


def test_merge_distributed_evaluator_state_loads_merged_state(monkeypatch) -> None:
    evaluator = _Evaluator()
    context = SimpleNamespace(enabled=True, world_size=2)

    monkeypatch.setattr(
        "adrf.protocol.runtime_support.all_gather_objects",
        lambda payload, distributed_context: [payload, {"updates": 2}],
    )

    merge_distributed_evaluator_state(evaluator, context)

    assert evaluator.loaded_state == {"updates": 3}
