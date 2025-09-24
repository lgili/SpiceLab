"""Common inductor entries for the component library."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..core.components import Inductor
from .registry import register_component


@dataclass(frozen=True)
class InductorEntry:
    slug: str
    value: str | float
    current: str
    dcr: str
    description: str

    def factory(self, ref: str, *, value: str | float | None = None) -> Inductor:
        return Inductor(ref, value if value is not None else self.value)

    def metadata(self) -> Mapping[str, object]:
        return {
            "value": self.value,
            "current_rating": self.current,
            "dcr": self.dcr,
            "description": self.description,
        }


_INDUCTORS = [
    InductorEntry(
        slug="power_10u_3a",
        value="10u",
        current="3A",
        dcr="45mΩ",
        description="Power inductor for buck converters",
    ),
    InductorEntry(
        slug="rf_100n_100ma",
        value="100n",
        current="100mA",
        dcr="1.2Ω",
        description="RF choke for bias tees",
    ),
]


def _register_defaults() -> None:
    for entry in _INDUCTORS:
        name = f"inductor.{entry.slug}"

        def _factory(
            ref: str, *, value: str | float | None = None, _entry: InductorEntry = entry
        ) -> Inductor:
            return _entry.factory(ref, value=value)

        register_component(
            name,
            _factory,
            category="inductor",
            metadata=entry.metadata(),
            overwrite=False,
        )


_register_defaults()

__all__ = ["_INDUCTORS"]
