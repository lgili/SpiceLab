"""Transistor component entries (BJT, MOSFET)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ..core.components import Component
from ..core.net import Port, PortRole
from .registry import register_component


class Mosfet(Component):
    """Basic 4-terminal MOSFET (drain, gate, source, bulk)."""

    def __init__(self, ref: str, model: str, params: str | None = None) -> None:
        super().__init__(ref=ref, value=model)
        self.params = params or ""
        self._ports = (
            Port(self, "d", PortRole.POSITIVE),
            Port(self, "g", PortRole.NODE),
            Port(self, "s", PortRole.NEGATIVE),
            Port(self, "b", PortRole.NODE),
        )

    def spice_card(self, net_of: Callable[[Port], str]) -> str:
        d, g, s, b = self.ports
        extra = f" {self.params}" if self.params else ""
        return f"M{self.ref} {net_of(d)} {net_of(g)} {net_of(s)} {net_of(b)} {self.value}{extra}"


class Bjt(Component):
    """Basic three-terminal BJT (collector, base, emitter)."""

    def __init__(self, ref: str, model: str, area: str | float | None = None) -> None:
        super().__init__(ref=ref, value=model)
        self.area = area
        self._ports = (
            Port(self, "c", PortRole.POSITIVE),
            Port(self, "b", PortRole.NODE),
            Port(self, "e", PortRole.NEGATIVE),
        )

    def spice_card(self, net_of: Callable[[Port], str]) -> str:
        c, b, e = self.ports
        area = f" {self.area}" if self.area is not None else ""
        return f"Q{self.ref} {net_of(c)} {net_of(b)} {net_of(e)} {self.value}{area}"


@dataclass(frozen=True)
class MosfetEntry:
    slug: str
    model_name: str
    description: str
    polarity: str
    default_params: str
    model_card: str

    def metadata(self) -> Mapping[str, object]:
        return {
            "model": self.model_name,
            "description": self.description,
            "polarity": self.polarity,
            "default_params": self.default_params,
            "model_card": self.model_card,
        }


@dataclass(frozen=True)
class BjtEntry:
    slug: str
    model_name: str
    description: str
    type: str  # NPN or PNP
    model_card: str

    def metadata(self) -> Mapping[str, object]:
        return {
            "model": self.model_name,
            "description": self.description,
            "type": self.type,
            "model_card": self.model_card,
        }


_MOSFETS = [
    MosfetEntry(
        slug="bss138",
        model_name="MBSS138",
        description="N-channel MOSFET, small-signal logic level",
        polarity="n-channel",
        default_params="",
        model_card=".model MBSS138 NMOS(Level=8 Vto=1.5 KP=1e-3 Lambda=0.02)",
    ),
    MosfetEntry(
        slug="irfz44n",
        model_name="MIRFZ44N",
        description="N-channel MOSFET, power switching",
        polarity="n-channel",
        default_params="L=100n W=10",
        model_card=".model MIRFZ44N NMOS(Vto=4 Rd=0.02 Rs=0.02 Rg=1 Cgd=1n Cgs=2n) ",
    ),
    MosfetEntry(
        slug="ao3401a",
        model_name="MAO3401A",
        description="P-channel MOSFET, logic-level load switch",
        polarity="p-channel",
        default_params="L=100n W=10",
        model_card=".model MAO3401A PMOS(Vto=-1.8 KP=0.8e-3 Rs=0.05 Rd=0.05)",
    ),
]

_BJTS = [
    BjtEntry(
        slug="2n3904",
        model_name="Q2N3904",
        description="NPN small-signal transistor",
        type="npn",
        model_card=".model Q2N3904 NPN(Is=6.734E-15 Bf=200 Vaf=100 Ikf=0.1)",
    ),
    BjtEntry(
        slug="2n3906",
        model_name="Q2N3906",
        description="PNP complement",
        type="pnp",
        model_card=".model Q2N3906 PNP(Is=1.0E-14 Bf=180 Vaf=60 Ikf=0.08)",
    ),
    BjtEntry(
        slug="bc547b",
        model_name="QBC547B",
        description="NPN low-noise transistor",
        type="npn",
        model_card=".model QBC547B NPN(Is=2.6E-14 Bf=290 Vaf=110 Ikf=0.05)",
    ),
]


def _make_mosfet_factory(entry: MosfetEntry) -> Callable[..., Mosfet]:
    def factory(ref: str, *, model: str | None = None, params: str | None = None) -> Mosfet:
        selected_model = model or entry.model_name
        combined = params if params is not None else entry.default_params
        return Mosfet(ref, selected_model, combined)

    return factory


def _make_bjt_factory(entry: BjtEntry) -> Callable[..., Bjt]:
    def factory(ref: str, *, model: str | None = None, area: str | float | None = None) -> Bjt:
        selected_model = model or entry.model_name
        return Bjt(ref, selected_model, area)

    return factory


def _register_defaults() -> None:
    for mosfet_entry in _MOSFETS:
        register_component(
            f"mosfet.{mosfet_entry.slug}",
            _make_mosfet_factory(mosfet_entry),
            category="mosfet",
            metadata=mosfet_entry.metadata(),
            overwrite=False,
        )

    for bjt_entry in _BJTS:
        register_component(
            f"bjt.{bjt_entry.slug}",
            _make_bjt_factory(bjt_entry),
            category="bjt",
            metadata=bjt_entry.metadata(),
            overwrite=False,
        )


_register_defaults()

__all__ = ["_MOSFETS", "_BJTS", "Mosfet", "Bjt"]
