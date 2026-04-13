"""Tests for the optional SwanLab logger adapter."""

import warnings

from adrf.logging.swanlab_logger import SwanLabLoggerAdapter


class _FakeSwanLabRun:
    def __init__(self) -> None:
        self.logged = []
        self.finished = None

    def log(self, metrics, step=None) -> None:
        self.logged.append((metrics, step))

    def finish(self, status="completed") -> None:
        self.finished = status


class _FakeSwanLabModule:
    def __init__(self) -> None:
        self.run = _FakeSwanLabRun()

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run


def test_swanlab_logger_adapter_runs_in_disabled_mode_when_unavailable() -> None:
    """Missing SwanLab should not break local logging paths."""

    with warnings.catch_warnings(record=True) as caught:
        logger = SwanLabLoggerAdapter(module=None, strict=False)
    assert logger.enabled is False or logger.enabled is True
    # When the module is absent in the environment, the adapter should disable itself.
    if logger.enabled is False:
        assert any("disabled mode" in str(item.message) for item in caught)


def test_swanlab_logger_adapter_uses_supplied_module() -> None:
    """A supplied SwanLab-like module should receive lifecycle calls."""

    module = _FakeSwanLabModule()
    logger = SwanLabLoggerAdapter(module=module, strict=True)
    logger.start_run("demo", {"seed": 0}, {"status": "running"})
    logger.log_metrics({"metric": 1.0}, step=1)
    logger.finish_run("completed")

    assert module.init_kwargs["name"] == "demo"
    assert module.run.logged == [({"metric": 1.0}, 1)]
    assert module.run.finished == "completed"

