"""Common resistor entries for the component library."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..core.components import Resistor
from .registry import register_component


@dataclass(frozen=True)
class ResistorEntry:
    slug: str
    value: str | float
    tolerance: str
    power: str
    description: str

    def factory(self, ref: str, *, value: str | float | None = None) -> Resistor:
        return Resistor(ref, value if value is not None else self.value)

    def metadata(self) -> Mapping[str, object]:
        return {
            "value": self.value,
            "tolerance": self.tolerance,
            "power": self.power,
            "description": self.description,
        }


_RESISTORS = [
    ResistorEntry(
        slug="e24.1k_1pct_0.25w",
        value="1k",
        tolerance="1%",
        power="0.25W",
        description="Metal film resistor for precision bias networks",
    ),
    ResistorEntry(
        slug="e24.4k7_5pct_0.5w",
        value="4.7k",
        tolerance="5%",
        power="0.5W",
        description="Carbon film resistor for general-purpose pull-ups",
    ),
    ResistorEntry(
        slug="shunt_0r01_1pct_1w",
        value="0.01",
        tolerance="1%",
        power="1W",
        description="Low-value current shunt resistor",
    ),
]


def _register_defaults() -> None:
    for entry in _RESISTORS:
        name = f"resistor.{entry.slug}"

        def _factory(
            ref: str, *, value: str | float | None = None, _entry: ResistorEntry = entry
        ) -> Resistor:
            return _entry.factory(ref, value=value)

        register_component(
            name,
            _factory,
            category="resistor",
            metadata=entry.metadata(),
            overwrite=False,
        )


_register_defaults()

__all__ = ["_RESISTORS"]
