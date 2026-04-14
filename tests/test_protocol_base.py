from types import SimpleNamespace

from adrf.protocol.base import BaseProtocol
from adrf.protocol.results import EvaluationSummary, TrainSummary


class _FakeProtocol(BaseProtocol):
    def __init__(self) -> None:
        self.train_contexts = []
        self.eval_contexts = []

    def train(self, context):
        self.train_contexts.append(context)
        return TrainSummary(
            num_train_batches=1,
            num_train_samples=2,
            metrics={"loss": 0.5},
        )

    def evaluate_context(self, context):
        self.eval_contexts.append(context)
        return EvaluationSummary(metrics={"image_auroc": 0.9})


class _LegacyProtocol(BaseProtocol):
    def __init__(self) -> None:
        self.train_runners = []
        self.eval_runners = []

    def train_epoch(self, runner):
        self.train_runners.append(runner)
        return {"num_train_batches": 3, "num_train_samples": 4, "loss": 0.25}

    def evaluate(self, runner):
        self.eval_runners.append(runner)
        return {"image_auroc": 0.7}


def test_base_protocol_run_reuses_one_context_and_serializes_typed_results(monkeypatch) -> None:
    protocol = _FakeProtocol()
    context = object()

    monkeypatch.setattr(protocol, "build_context", lambda runner: context)

    results = protocol.run(SimpleNamespace())

    assert results == {
        "train": {"num_train_batches": 1, "num_train_samples": 2, "loss": 0.5},
        "evaluation": {"image_auroc": 0.9},
    }
    assert protocol.train_contexts == [context]
    assert protocol.eval_contexts == [context]


def test_base_protocol_train_epoch_and_evaluate_keep_public_contract(monkeypatch) -> None:
    protocol = _FakeProtocol()
    context = object()

    monkeypatch.setattr(protocol, "build_context", lambda runner: context)

    assert protocol.train_epoch(SimpleNamespace()) == {
        "num_train_batches": 1,
        "num_train_samples": 2,
        "loss": 0.5,
    }
    assert protocol.evaluate(SimpleNamespace()) == {"image_auroc": 0.9}


def test_base_protocol_legacy_runner_overrides_remain_instantiable_and_used_by_run(monkeypatch) -> None:
    protocol = _LegacyProtocol()
    runner = SimpleNamespace()

    monkeypatch.setattr(protocol, "build_context", lambda runner: (_ for _ in ()).throw(AssertionError("build_context should not be used")))

    assert protocol.run(runner) == {
        "train": {"num_train_batches": 3, "num_train_samples": 4, "loss": 0.25},
        "evaluation": {"image_auroc": 0.7},
    }
    assert protocol.train_runners == [runner]
    assert protocol.eval_runners == [runner]
