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
    # Signal diodes
    DiodeEntry(
        part_number="1N4148",
        model="D1N4148",
        description="Small-signal switching diode",
        model_card=".model D1N4148 D(Is=2.52e-9 Rs=0.568 N=1.906 Cjo=4p M=0.333 Vj=0.75)",
    ),
    DiodeEntry(
        part_number="1N914",
        model="D1N914",
        description="Small-signal switching diode (equivalent to 1N4148)",
        model_card=".model D1N914 D(Is=2.52e-9 Rs=0.568 N=1.906 Cjo=4p M=0.333 Vj=0.75)",
    ),
    # Rectifier diodes
    DiodeEntry(
        part_number="1N4001",
        model="D1N4001",
        description="Rectifier diode 1A/50V",
        model_card=".model D1N4001 D(Is=14.11n Rs=0.05 N=1.984 Cjo=25p M=0.333 Bv=50 Ibv=100u)",
    ),
    DiodeEntry(
        part_number="1N4004",
        model="D1N4004",
        description="Rectifier diode 1A/400V",
        model_card=".model D1N4004 D(Is=14.11n Rs=0.04 N=1.984 Cjo=22p M=0.333 Bv=400 Ibv=100u)",
    ),
    DiodeEntry(
        part_number="1N4007",
        model="D1N4007",
        description="Rectifier diode 1A/1000V",
        model_card=".model D1N4007 D(Is=14.11n Rs=0.033 N=1.984 Bv=1000 Ibv=100u)",
    ),
    # Schottky diodes
    DiodeEntry(
        part_number="1N5817",
        model="D1N5817",
        description="Schottky diode 1A/20V",
        model_card=".model D1N5817 D(Is=31.7u Rs=0.051 N=1.05 Cjo=150p M=0.5 Bv=20 Ibv=1m)",
    ),
    DiodeEntry(
        part_number="1N5818",
        model="D1N5818",
        description="Schottky diode 1A/30V",
        model_card=".model D1N5818 D(Is=28.4u Rs=0.043 N=1.05 Cjo=145p M=0.5 Bv=30 Ibv=1m)",
    ),
    DiodeEntry(
        part_number="1N5819",
        model="D1N5819",
        description="Schottky diode 1A/40V",
        model_card=".model D1N5819 D(Is=26.5u Rs=0.036 N=1.05 Cjo=140p M=0.5 Bv=40 Ibv=1m)",
    ),
    DiodeEntry(
        part_number="BAT54",
        model="DBAT54",
        description="Schottky barrier diode 30V/200mA",
        model_card=".model DBAT54 D(Is=1u Rs=0.8 N=1.03 Cjo=10p Bv=30 Ibv=1u)",
    ),
    # Zener diodes
    DiodeEntry(
        part_number="1N4728A",
        model="DZ3V3",
        description="Zener diode 3.3V/1W",
        model_card=".model DZ3V3 D(Is=1.8e-9 Rs=15 N=1.5 Bv=3.3 Ibv=5m)",
    ),
    DiodeEntry(
        part_number="1N4733A",
        model="DZ5V1",
        description="Zener diode 5.1V/1W",
        model_card=".model DZ5V1 D(Is=3.0e-9 Rs=10 N=1.5 Bv=5.1 Ibv=5m)",
    ),
    DiodeEntry(
        part_number="1N4742A",
        model="DZ12V",
        description="Zener diode 12V/1W",
        model_card=".model DZ12V D(Is=3.0e-9 Rs=5 N=1.5 Bv=12 Ibv=5m)",
    ),
    DiodeEntry(
        part_number="1N4744A",
        model="DZ15V",
        description="Zener diode 15V/1W",
        model_card=".model DZ15V D(Is=3.0e-9 Rs=4 N=1.5 Bv=15 Ibv=5m)",
    ),
    DiodeEntry(
        part_number="BZX55C5V1",
        model="DZBZX5V1",
        description="Zener diode 5.1V/500mW",
        model_card=".model DZBZX5V1 D(Is=2.5e-9 Rs=12 N=1.5 Bv=5.1 Ibv=1m)",
    ),
    # LEDs (simplified models)
    DiodeEntry(
        part_number="LED_RED",
        model="DLED_RED",
        description="Red LED (Vf~1.8V)",
        model_card=".model DLED_RED D(Is=1e-20 Rs=5 N=2.0 Eg=1.95)",
    ),
    DiodeEntry(
        part_number="LED_GREEN",
        model="DLED_GREEN",
        description="Green LED (Vf~2.1V)",
        model_card=".model DLED_GREEN D(Is=1e-21 Rs=5 N=2.2 Eg=2.26)",
    ),
    DiodeEntry(
        part_number="LED_BLUE",
        model="DLED_BLUE",
        description="Blue LED (Vf~3.2V)",
        model_card=".model DLED_BLUE D(Is=1e-24 Rs=5 N=2.5 Eg=3.4)",
    ),
    DiodeEntry(
        part_number="LED_WHITE",
        model="DLED_WHITE",
        description="White LED (Vf~3.3V)",
        model_card=".model DLED_WHITE D(Is=1e-24 Rs=4 N=2.5 Eg=3.4)",
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
