"""Common capacitor entries for the component library."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..core.components import Capacitor
from .registry import register_component


@dataclass(frozen=True)
class CapacitorEntry:
    slug: str
    value: str | float
    dielectric: str
    voltage: str
    description: str

    def factory(self, ref: str, *, value: str | float | None = None) -> Capacitor:
        return Capacitor(ref, value if value is not None else self.value)

    def metadata(self) -> Mapping[str, object]:
        return {
            "value": self.value,
            "dielectric": self.dielectric,
            "voltage": self.voltage,
            "description": self.description,
        }


_CAPACITORS = [
    CapacitorEntry(
        slug="ceramic_100n_50v_x7r",
        value="100n",
        dielectric="X7R",
        voltage="50V",
        description="Decoupling capacitor for MCU rails",
    ),
    CapacitorEntry(
        slug="electrolytic_10u_25v",
        value="10u",
        dielectric="Aluminium electrolytic",
        voltage="25V",
        description="Bulk storage capacitor for DC rails",
    ),
    CapacitorEntry(
        slug="film_1u_250v",
        value="1u",
        dielectric="Polypropylene film",
        voltage="250V",
        description="Film capacitor for audio coupling and snubbers",
    ),
]


def _register_defaults() -> None:
    for entry in _CAPACITORS:
        name = f"capacitor.{entry.slug}"

        def _factory(
            ref: str, *, value: str | float | None = None, _entry: CapacitorEntry = entry
        ) -> Capacitor:
            return _entry.factory(ref, value=value)

        register_component(
            name,
            _factory,
            category="capacitor",
            metadata=entry.metadata(),
            overwrite=False,
        )


_register_defaults()

__all__ = ["_CAPACITORS"]
