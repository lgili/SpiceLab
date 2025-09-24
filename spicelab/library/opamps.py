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
    OpAmpEntry(
        slug="ideal",
        description="Ideal op-amp with very high gain",
        variant="ideal",
        gain="1e6",
    ),
    OpAmpEntry(
        slug="lm741",
        description="LM741 macro-model",
        variant="subckt",
        subckt_name="LM741",
        includes=(str(_DATA_ROOT.joinpath("lm741.sub")),),
    ),
    OpAmpEntry(
        slug="tl081",
        description="TL081 FET-input op-amp",
        variant="subckt",
        subckt_name="TL081",
        includes=(str(_DATA_ROOT.joinpath("tl081.sub")),),
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
