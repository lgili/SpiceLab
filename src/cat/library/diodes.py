"""Common diode component factories."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..core.components import Diode
from .registry import register_component


@dataclass(frozen=True)
class DiodeEntry:
    part_number: str
    model: str
    description: str
    model_card: str | None = None

    def factory(self, ref: str, *, model: str | None = None) -> Diode:
        return Diode(ref, model or self.model)

    def metadata(self) -> Mapping[str, object]:
        data: dict[str, object] = {"description": self.description, "model": self.model}
        if self.model_card:
            data["model_card"] = self.model_card
        return data


_COMMON_DIODES = [
    DiodeEntry(
        part_number="1N4148",
        model="D1N4148",
        description="Small-signal switching diode",
        model_card=".model D1N4148 D(Is=2.52e-9 Rs=0.568 N=1.906 Cjo=4p M=0.333 Vj=0.75)",
    ),
    DiodeEntry(
        part_number="1N4007",
        model="D1N4007",
        description="Rectifier diode 1A/1000V",
        model_card=".model D1N4007 D(Is=14.11n Rs=0.033 N=1.984 Cjo=21p M=0.333 Eg=1.11)",
    ),
]


def _register_defaults() -> None:
    for entry in _COMMON_DIODES:
        name = f"diode.{entry.part_number.lower()}"

        def _factory(ref: str, *, model: str | None = None, _entry: DiodeEntry = entry) -> Diode:
            return _entry.factory(ref, model=model)

        register_component(
            name,
            _factory,
            category="diode",
            metadata=entry.metadata(),
            overwrite=False,
        )


_register_defaults()

__all__ = ["_COMMON_DIODES"]
