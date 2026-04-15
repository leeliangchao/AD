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


class _PhaseTrackingProtocol(_FakeProtocol):
    def __init__(self) -> None:
        super().__init__()
        self.train_builder_runners = []
        self.evaluate_builder_runners = []

    def build_context(self, runner):
        del runner
        raise AssertionError("typed run() should not call build_context().")

    def build_train_context(self, runner):
        self.train_builder_runners.append(runner)
        return "train-context"

    def build_evaluate_context(self, runner):
        self.evaluate_builder_runners.append(runner)
        return "evaluate-context"


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


class _MixedProtocol(BaseProtocol):
    def train(self, context):
        return TrainSummary(
            num_train_batches=2,
            num_train_samples=3,
            metrics={"loss": 0.4},
        )

    def train_epoch(self, runner):
        return {"num_train_batches": 99, "num_train_samples": 100, "loss": 9.9}

    def evaluate_context(self, context):
        return EvaluationSummary(metrics={"image_auroc": 0.8})


class _BuildContextOnlyTypedProtocol(BaseProtocol):
    def __init__(self) -> None:
        self.build_context_calls = 0
        self.train_context_indexes = []
        self.evaluate_context_indexes = []

    def build_context(self, runner):
        self.build_context_calls += 1
        return SimpleNamespace(
            call_index=self.build_context_calls,
            train_loader=f"runner-train-{self.build_context_calls}",
            test_loader=f"runner-test-{self.build_context_calls}",
            runner=runner,
        )

    def train(self, context):
        self.train_context_indexes.append(context.call_index)
        return TrainSummary(
            num_train_batches=context.call_index,
            num_train_samples=context.call_index + 1,
            metrics={"loss": float(context.call_index)},
        )

    def evaluate_context(self, context):
        self.evaluate_context_indexes.append(context.call_index)
        return EvaluationSummary(metrics={"image_auroc": float(context.call_index)})


def test_base_protocol_run_uses_phase_specific_builders_for_typed_results() -> None:
    protocol = _PhaseTrackingProtocol()
    runner = SimpleNamespace()

    results = protocol.run(runner)

    assert results == {
        "train": {"num_train_batches": 1, "num_train_samples": 2, "loss": 0.5},
        "evaluation": {"image_auroc": 0.9},
    }
    assert protocol.train_builder_runners == [runner]
    assert protocol.evaluate_builder_runners == [runner]
    assert protocol.train_contexts == ["train-context"]
    assert protocol.eval_contexts == ["evaluate-context"]


def test_base_protocol_typed_defaults_respect_build_context_override() -> None:
    protocol = _BuildContextOnlyTypedProtocol()
    runner = SimpleNamespace(
        datamodule=SimpleNamespace(
            train_dataloader=lambda: "runner-train-loader",
            test_dataloader=lambda: "runner-test-loader",
        ),
        normality=object(),
        evidence=object(),
        evaluator=object(),
    )

    train_summary = protocol.train_epoch(runner)
    evaluation_summary = protocol.evaluate(runner)
    run_summary = protocol.run(runner)

    assert train_summary == {
        "num_train_batches": 1,
        "num_train_samples": 2,
        "loss": 1.0,
    }
    assert evaluation_summary == {"image_auroc": 2.0}
    assert run_summary == {
        "train": {"num_train_batches": 3, "num_train_samples": 4, "loss": 3.0},
        "evaluation": {"image_auroc": 4.0},
    }
    assert protocol.build_context_calls == 4
    assert protocol.train_context_indexes == [1, 3]
    assert protocol.evaluate_context_indexes == [2, 4]


def test_base_protocol_train_epoch_and_evaluate_keep_public_contract(monkeypatch) -> None:
    protocol = _FakeProtocol()
    context = object()

    monkeypatch.setattr(protocol, "build_train_context", lambda runner: context)
    monkeypatch.setattr(protocol, "build_evaluate_context", lambda runner: context)

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


def test_base_protocol_typed_train_epoch_supports_train_only_runner() -> None:
    protocol = _FakeProtocol()
    train_loader = object()
    runner = SimpleNamespace(
        datamodule=SimpleNamespace(
            train_dataloader=lambda: train_loader,
        ),
        representation=object(),
        normality=object(),
    )

    assert protocol.train_epoch(runner) == {
        "num_train_batches": 1,
        "num_train_samples": 2,
        "loss": 0.5,
    }


def test_base_protocol_typed_evaluate_supports_test_only_runner() -> None:
    protocol = _FakeProtocol()
    test_loader = object()
    runner = SimpleNamespace(
        datamodule=SimpleNamespace(
            test_dataloader=lambda: test_loader,
        ),
        representation=object(),
        normality=object(),
        evidence=object(),
        evaluator=object(),
    )

    assert protocol.evaluate(runner) == {"image_auroc": 0.9}


def test_base_protocol_run_rejects_mixed_mode_protocol() -> None:
    protocol = _MixedProtocol()
    runner = SimpleNamespace(
        datamodule=SimpleNamespace(
            train_dataloader=lambda: object(),
            test_dataloader=lambda: object(),
        ),
        representation=object(),
        normality=object(),
        evidence=object(),
        evaluator=object(),
    )

    try:
        protocol.run(runner)
    except RuntimeError as exc:
        assert "mixed" in str(exc).lower()
    else:
        raise AssertionError("mixed protocol should raise RuntimeError")
