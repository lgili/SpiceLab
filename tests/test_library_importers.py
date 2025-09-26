from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor
from spicelab.core.net import Port
from spicelab.core.typing import HasPorts
from spicelab.library import importers
from spicelab.library.registry import ComponentSpec
from spicelab.library.utils import _coerce_to_lines, apply_metadata_to_circuit


class DummyRegistry:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def register(
        self,
        name: str,
        factory: Any,
        *,
        category: Any = None,
        metadata: Any = None,
        overwrite: bool = False,
    ) -> None:
        self.calls.append(
            {
                "name": name,
                "factory": factory,
                "category": category,
                "metadata": metadata,
                "overwrite": overwrite,
            }
        )


@pytest.fixture(autouse=True)
def patch_registry(monkeypatch: pytest.MonkeyPatch) -> DummyRegistry:
    reg = DummyRegistry()
    monkeypatch.setattr(importers, "register_component", reg.register)
    return reg


def test_import_catalog_from_mapping_uses_factory_builder(patch_registry: DummyRegistry) -> None:
    entries = [{"name": "R1", "value": "1k", "category": "res", "meta": {"pkg": "0402"}}]

    def builder(entry: Mapping[str, Any]) -> Callable[[], Resistor]:
        def factory() -> Resistor:
            return Resistor(str(entry["name"]), str(entry["value"]))

        return factory

    importers.import_catalog_from_mapping(
        entries,
        category_key="category",
        metadata_key="meta",
        factory_builder=builder,
        overwrite=True,
    )
    call = patch_registry.calls[0]
    assert call["name"] == "R1"
    component = call["factory"]()
    assert isinstance(component, Resistor)
    assert component.value == "1k"
    assert call["category"] == "res"
    assert call["metadata"] == {"pkg": "0402"}
    assert call["overwrite"] is True


def test_import_catalog_from_json(tmp_path: Path, patch_registry: DummyRegistry) -> None:
    data = [{"name": "C1", "value": "3.3"}]
    file = tmp_path / "caps.json"
    file.write_text(json.dumps(data), encoding="utf-8")

    def builder(entry: Mapping[str, Any]) -> Callable[[], Resistor]:
        def factory() -> Resistor:
            return Resistor(str(entry["name"]), str(entry.get("value", "")))

        return factory

    importers.import_catalog_from_json(file, factory_builder=builder)
    call = patch_registry.calls[0]
    assert call["name"] == "C1"
    factory = call["factory"]
    component = factory()
    assert isinstance(component, Resistor)
    assert component.value == "3.3"


def test_import_catalog_from_csv(tmp_path: Path, patch_registry: DummyRegistry) -> None:
    csv_path = tmp_path / "regs.csv"
    csv_path.write_text("name,factory\nX,lambda value: value\n", encoding="utf-8")

    def builder(entry: Mapping[str, Any]) -> Callable[[], Resistor]:
        def factory() -> Resistor:
            return Resistor(str(entry["name"]), "default")

        return factory

    importers.import_catalog_from_csv(csv_path, factory_builder=builder)
    call = patch_registry.calls[0]
    component = call["factory"]()
    assert isinstance(component, Resistor)
    assert component.ref == "X"


def test_ensure_path_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(importers.ImportError):
        importers.import_catalog_from_json(missing)


def test_coerce_to_lines_variants() -> None:
    assert _coerce_to_lines(None) == []
    assert _coerce_to_lines(".include lib.asc") == [".include lib.asc"]
    assert _coerce_to_lines([".model X", None, 1]) == [".model X", "1"]


def test_apply_metadata_to_circuit() -> None:
    circuit = Circuit("meta")
    metadata = {
        "include": "lib.lib",
        "model_card": [".model RX resistor"],
        "extras": [".param temp=27"],
    }
    spec = ComponentSpec(name="R1", factory=lambda: Resistor("R1", "1k"), metadata=metadata)
    added = apply_metadata_to_circuit(
        circuit,
        spec,
        directive_keys=("extras",),
    )
    assert any(line.startswith(".include") for line in added)
    assert any(".model" in line for line in added)
    assert any(".param" in line for line in added)


def test_has_ports_protocol() -> None:
    class Dummy(HasPorts):
        def __init__(self) -> None:
            owner = Resistor("R1", "1k")
            self._ports: tuple[Port, ...] = tuple(owner.ports)

        @property
        def ports(self) -> tuple[Port, ...]:
            return self._ports

    dummy = Dummy()
    assert isinstance(dummy, HasPorts)
