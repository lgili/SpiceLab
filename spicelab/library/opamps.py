"""Operational amplifier entries for the component library."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.resources import files

from ..core.components import OpAmpIdeal
from ..core.net import Port, PortRole
from .registry import register_component


class OpAmpSubckt(OpAmpIdeal):
    """Subckt-based op-amp wrapper with supply pins."""

    def __init__(self, ref: str, subckt_name: str) -> None:
        # Use high gain for ideal placeholder (not used)
        super().__init__(ref=ref, gain="1e6")
        self.subckt = subckt_name
        self._ports = (
            Port(self, "non", PortRole.POSITIVE),
            Port(self, "inv", PortRole.NEGATIVE),
            Port(self, "out", PortRole.NODE),
            Port(self, "v+", PortRole.POSITIVE),
            Port(self, "v-", PortRole.NEGATIVE),
        )

    def spice_card(self, net_of: Callable[[Port], str]) -> str:
        non, inv, out, vp, vn = self.ports
        return (
            f"X{self.ref} {net_of(non)} {net_of(inv)} {net_of(out)} "
            f"{net_of(vp)} {net_of(vn)} {self.subckt}"
        )


@dataclass(frozen=True)
class OpAmpEntry:
    slug: str
    description: str
    variant: str  # "ideal" or "subckt"
    gain: str | float | None = None
    subckt_name: str | None = None
    model_card: str | None = None
    includes: tuple[str, ...] | None = None
    extra: Mapping[str, object] | None = None

    def metadata(self) -> Mapping[str, object]:
        data: dict[str, object] = {
            "description": self.description,
            "variant": self.variant,
        }
        if self.model_card:
            data["model_card"] = self.model_card
        if self.subckt_name:
            data["subckt"] = self.subckt_name
        if self.gain is not None:
            data["gain"] = self.gain
        if self.includes:
            if len(self.includes) == 1:
                data["include"] = self.includes[0]
            else:
                data["include"] = list(self.includes)
        if self.extra:
            data.update(self.extra)
        return data


_DATA_ROOT = files("spicelab.library.data.opamps")

_OPAMPS = [
    # Ideal op-amp
    OpAmpEntry(
        slug="ideal",
        description="Ideal op-amp with very high gain",
        variant="ideal",
        gain="1e6",
    ),
    # General purpose op-amps
    OpAmpEntry(
        slug="lm741",
        description="LM741 general-purpose op-amp (classic)",
        variant="subckt",
        subckt_name="LM741",
        includes=(str(_DATA_ROOT.joinpath("lm741.sub")),),
        extra={"gbw": "1MHz", "slew_rate": "0.5V/us", "input_type": "bipolar"},
    ),
    OpAmpEntry(
        slug="lm358",
        description="LM358 dual low-power op-amp, single supply",
        variant="subckt",
        subckt_name="LM358",
        includes=(str(_DATA_ROOT.joinpath("lm358.sub")),),
        extra={"gbw": "1MHz", "slew_rate": "0.3V/us", "input_type": "bipolar", "supply": "single"},
    ),
    OpAmpEntry(
        slug="lm324",
        description="LM324 quad low-power op-amp, single supply",
        variant="subckt",
        subckt_name="LM324",
        includes=(str(_DATA_ROOT.joinpath("lm324.sub")),),
        extra={"gbw": "1MHz", "slew_rate": "0.5V/us", "input_type": "bipolar", "supply": "single"},
    ),
    # JFET-input op-amps (high impedance)
    OpAmpEntry(
        slug="tl081",
        description="TL081 JFET-input op-amp, low noise",
        variant="subckt",
        subckt_name="TL081",
        includes=(str(_DATA_ROOT.joinpath("tl081.sub")),),
        extra={"gbw": "3MHz", "slew_rate": "13V/us", "input_type": "jfet"},
    ),
    OpAmpEntry(
        slug="tl072",
        description="TL072 dual JFET-input op-amp, low noise, audio-grade",
        variant="subckt",
        subckt_name="TL072",
        includes=(str(_DATA_ROOT.joinpath("tl072.sub")),),
        extra={"gbw": "3MHz", "slew_rate": "13V/us", "input_type": "jfet"},
    ),
    # Audio op-amps
    OpAmpEntry(
        slug="ne5532",
        description="NE5532 dual low-noise op-amp, audio applications",
        variant="subckt",
        subckt_name="NE5532",
        includes=(str(_DATA_ROOT.joinpath("ne5532.sub")),),
        extra={"gbw": "10MHz", "slew_rate": "9V/us", "input_type": "bipolar", "noise": "5nV/sqrt(Hz)"},
    ),
    OpAmpEntry(
        slug="opa2134",
        description="OPA2134 dual FET-input audio op-amp, low distortion",
        variant="subckt",
        subckt_name="OPA2134",
        includes=(str(_DATA_ROOT.joinpath("opa2134.sub")),),
        extra={"gbw": "8MHz", "slew_rate": "20V/us", "input_type": "fet", "thd": "0.00008%"},
    ),
    OpAmpEntry(
        slug="lm386",
        description="LM386 low voltage audio power amplifier",
        variant="subckt",
        subckt_name="LM386",
        includes=(str(_DATA_ROOT.joinpath("lm386.sub")),),
        extra={"voltage_gain": "20-200", "power_output": "0.5W", "supply": "4-12V"},
    ),
    # Precision op-amps
    OpAmpEntry(
        slug="op07",
        description="OP07 ultra-low offset voltage precision op-amp",
        variant="subckt",
        subckt_name="OP07",
        includes=(str(_DATA_ROOT.joinpath("op07.sub")),),
        extra={"vos": "10uV", "drift": "0.2uV/C", "input_type": "bipolar"},
    ),
    # Low-power/micropower op-amps
    OpAmpEntry(
        slug="mcp6001",
        description="MCP6001 1MHz low-power rail-to-rail op-amp",
        variant="subckt",
        subckt_name="MCP6001",
        includes=(str(_DATA_ROOT.joinpath("mcp6001.sub")),),
        extra={"gbw": "1MHz", "supply": "1.8-6V", "quiescent": "100uA", "rail_to_rail": True},
    ),
    # Instrumentation amplifiers
    OpAmpEntry(
        slug="ad8221",
        description="AD8221 precision instrumentation amplifier",
        variant="subckt",
        subckt_name="AD8221",
        includes=(str(_DATA_ROOT.joinpath("ad8221.sub")),),
        extra={"cmrr": "80dB", "gain_accuracy": "0.02%", "type": "instrumentation"},
    ),
    OpAmpEntry(
        slug="ina128",
        description="INA128 precision low-power instrumentation amplifier",
        variant="subckt",
        subckt_name="INA128",
        includes=(str(_DATA_ROOT.joinpath("ina128.sub")),),
        extra={"cmrr": "120dB", "vos": "50uV", "type": "instrumentation"},
    ),
]


def _make_opamp_factory(entry: OpAmpEntry) -> Callable[..., OpAmpIdeal]:
    if entry.variant == "ideal":
        default_gain = entry.gain if entry.gain is not None else "1e6"

        def _ideal_factory(ref: str, *, gain: str | float | None = None) -> OpAmpIdeal:
            return OpAmpIdeal(ref, gain=gain or default_gain)

        return _ideal_factory

    subckt = entry.subckt_name or entry.slug.upper()

    def _subckt_factory(ref: str, *, subckt_name: str | None = None) -> OpAmpIdeal:
        return OpAmpSubckt(ref, subckt_name or subckt)

    return _subckt_factory


def _register_defaults() -> None:
    for entry in _OPAMPS:
        register_component(
            f"opamp.{entry.slug}",
            _make_opamp_factory(entry),
            category="opamp",
            metadata=entry.metadata(),
            overwrite=False,
        )


_register_defaults()

__all__ = ["_OPAMPS", "OpAmpSubckt"]
