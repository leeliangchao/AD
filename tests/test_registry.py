"""Tests for the registry and config utilities."""

from pathlib import Path

import pytest

from adrf.registry.registry import Registry
from adrf.utils.config import instantiate_component, load_yaml_config


class DummyComponent:
    """Small helper class for config instantiation tests."""

    def __init__(self, value: int, enabled: bool = False) -> None:
        self.value = value
        self.enabled = enabled


def test_registry_registers_and_retrieves_objects() -> None:
    """The registry should support group/name registration and lookup."""

    registry = Registry()

    registry.register("representation", "dummy", DummyComponent)

    assert registry.exists("representation", "dummy") is True
    assert registry.get("representation", "dummy") is DummyComponent
    assert registry.list_available("representation") == ["dummy"]


def test_registry_rejects_duplicate_names_within_group() -> None:
    """Registering the same group/name twice should raise an error."""

    registry = Registry()
    registry.register("representation", "dummy", DummyComponent)

    with pytest.raises(KeyError, match="representation"):
        registry.register("representation", "dummy", DummyComponent)


def test_registry_get_raises_for_unknown_entry() -> None:
    """Unknown entries should raise a clear lookup error."""

    registry = Registry()

    with pytest.raises(KeyError, match="missing"):
        registry.get("representation", "missing")


def test_load_yaml_config_and_instantiate_component(tmp_path: Path) -> None:
    """YAML configs should be loaded into dicts and instantiated via the registry."""

    config_path = tmp_path / "component.yaml"
    config_path.write_text(
        "\n".join(
            [
                "representation:",
                "  name: dummy",
                "  params:",
                "    value: 7",
                "    enabled: true",
            ]
        ),
        encoding="utf-8",
    )

    registry = Registry()
    registry.register("representation", "dummy", DummyComponent)

    config = load_yaml_config(config_path)
    component = instantiate_component(config["representation"], registry=registry, group="representation")

    assert config == {"representation": {"name": "dummy", "params": {"value": 7, "enabled": True}}}
    assert isinstance(component, DummyComponent)
    assert component.value == 7
    assert component.enabled is True
