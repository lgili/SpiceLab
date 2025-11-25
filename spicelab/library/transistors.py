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
    # N-channel small signal
    MosfetEntry(
        slug="2n7000",
        model_name="M2N7000",
        description="N-channel MOSFET 60V/200mA, small-signal",
        polarity="n-channel",
        default_params="",
        model_card=".model M2N7000 NMOS(Vto=2.1 KP=0.1 Lambda=0.04 Rd=0.2 Rs=0.2 Cgd=20p Cgs=30p)",
    ),
    MosfetEntry(
        slug="bss138",
        model_name="MBSS138",
        description="N-channel MOSFET, small-signal logic level",
        polarity="n-channel",
        default_params="",
        model_card=".model MBSS138 NMOS(Vto=1.5 KP=1e-3 Lambda=0.02 Rd=0.3 Rs=0.3)",
    ),
    MosfetEntry(
        slug="bs170",
        model_name="MBS170",
        description="N-channel MOSFET 60V/500mA, small-signal",
        polarity="n-channel",
        default_params="",
        model_card=".model MBS170 NMOS(Vto=2.1 KP=0.08 Lambda=0.03 Rd=0.5 Rs=0.5 Cgd=30p Cgs=50p)",
    ),
    # N-channel power
    MosfetEntry(
        slug="irf540n",
        model_name="MIRF540N",
        description="N-channel MOSFET 100V/33A, power switching",
        polarity="n-channel",
        default_params="",
        model_card=".model MIRF540N NMOS(Vto=4 KP=20 Rd=0.044)",
    ),
    MosfetEntry(
        slug="irfz44n",
        model_name="MIRFZ44N",
        description="N-channel MOSFET 55V/49A, power switching",
        polarity="n-channel",
        default_params="",
        model_card=".model MIRFZ44N NMOS(Vto=4 KP=30 Rd=0.028)",
    ),
    MosfetEntry(
        slug="irf3205",
        model_name="MIRF3205",
        description="N-channel MOSFET 55V/110A, power switching",
        polarity="n-channel",
        default_params="",
        model_card=".model MIRF3205 NMOS(Vto=3.5 KP=80 Rd=0.008)",
    ),
    MosfetEntry(
        slug="irlz44n",
        model_name="MIRLZ44N",
        description="N-channel MOSFET 55V/47A, logic-level gate",
        polarity="n-channel",
        default_params="",
        model_card=".model MIRLZ44N NMOS(Vto=2.0 KP=35 Rd=0.022)",
    ),
    # P-channel small signal
    MosfetEntry(
        slug="bs250",
        model_name="MBS250",
        description="P-channel MOSFET -45V/-180mA, small-signal",
        polarity="p-channel",
        default_params="",
        model_card=".model MBS250 PMOS(Vto=-3.0 KP=0.04 Rd=2.0)",
    ),
    # P-channel power
    MosfetEntry(
        slug="irf9540n",
        model_name="MIRF9540N",
        description="P-channel MOSFET -100V/-23A, power switching",
        polarity="p-channel",
        default_params="",
        model_card=".model MIRF9540N PMOS(Vto=-4 KP=8 Rd=0.117)",
    ),
    MosfetEntry(
        slug="ao3401a",
        model_name="MAO3401A",
        description="P-channel MOSFET -30V/-4A, logic-level load switch",
        polarity="p-channel",
        default_params="",
        model_card=".model MAO3401A PMOS(Vto=-1.8 KP=0.8e-3 Rd=0.05)",
    ),
    MosfetEntry(
        slug="irf4905",
        model_name="MIRF4905",
        description="P-channel MOSFET -55V/-74A, power switching",
        polarity="p-channel",
        default_params="",
        model_card=".model MIRF4905 PMOS(Vto=-3.5 KP=50 Rd=0.02)",
    ),
]

_BJTS = [
    # NPN general purpose
    BjtEntry(
        slug="2n2222",
        model_name="Q2N2222",
        description="NPN general-purpose transistor 40V/800mA",
        type="npn",
        model_card=".model Q2N2222 NPN(Is=14.34E-15 Bf=256 Vaf=74 Ikf=0.3)",
    ),
    BjtEntry(
        slug="2n2222a",
        model_name="Q2N2222A",
        description="NPN general-purpose transistor 40V/800mA (improved)",
        type="npn",
        model_card=".model Q2N2222A NPN(Is=14.34E-15 Bf=300 Vaf=80 Ikf=0.35)",
    ),
    BjtEntry(
        slug="2n3904",
        model_name="Q2N3904",
        description="NPN small-signal transistor 40V/200mA",
        type="npn",
        model_card=".model Q2N3904 NPN(Is=6.734E-15 Bf=416 Vaf=74 Ikf=0.067)",
    ),
    BjtEntry(
        slug="2n4401",
        model_name="Q2N4401",
        description="NPN small-signal transistor 40V/600mA",
        type="npn",
        model_card=".model Q2N4401 NPN(Is=26E-15 Bf=400 Vaf=100 Ikf=0.3)",
    ),
    BjtEntry(
        slug="bc547b",
        model_name="QBC547B",
        description="NPN low-noise transistor 45V/100mA",
        type="npn",
        model_card=".model QBC547B NPN(Is=2.6E-14 Bf=290 Vaf=110 Ikf=0.05)",
    ),
    BjtEntry(
        slug="bc548",
        model_name="QBC548",
        description="NPN general-purpose transistor 30V/100mA",
        type="npn",
        model_card=".model QBC548 NPN(Is=2.5E-14 Bf=280 Vaf=100 Ikf=0.04)",
    ),
    BjtEntry(
        slug="mpsa06",
        model_name="QMPSA06",
        description="NPN small-signal transistor 80V/500mA",
        type="npn",
        model_card=".model QMPSA06 NPN(Is=10E-15 Bf=300 Vaf=120 Ikf=0.2)",
    ),
    # PNP general purpose
    BjtEntry(
        slug="2n2907",
        model_name="Q2N2907",
        description="PNP general-purpose transistor -60V/-600mA",
        type="pnp",
        model_card=".model Q2N2907 PNP(Is=650.6E-18 Bf=232 Vaf=116 Ikf=0.3)",
    ),
    BjtEntry(
        slug="2n3906",
        model_name="Q2N3906",
        description="PNP small-signal transistor -40V/-200mA",
        type="pnp",
        model_card=".model Q2N3906 PNP(Is=1.41E-15 Bf=232 Vaf=116 Ikf=0.043)",
    ),
    BjtEntry(
        slug="2n4403",
        model_name="Q2N4403",
        description="PNP small-signal transistor -40V/-600mA",
        type="pnp",
        model_card=".model Q2N4403 PNP(Is=15E-15 Bf=350 Vaf=90 Ikf=0.25)",
    ),
    BjtEntry(
        slug="bc557",
        model_name="QBC557",
        description="PNP low-noise transistor -45V/-100mA",
        type="pnp",
        model_card=".model QBC557 PNP(Is=2.4E-14 Bf=270 Vaf=100 Ikf=0.04)",
    ),
    BjtEntry(
        slug="bc558",
        model_name="QBC558",
        description="PNP general-purpose transistor -30V/-100mA",
        type="pnp",
        model_card=".model QBC558 PNP(Is=2.3E-14 Bf=260 Vaf=90 Ikf=0.035)",
    ),
    BjtEntry(
        slug="mpsa56",
        model_name="QMPSA56",
        description="PNP small-signal transistor -80V/-500mA",
        type="pnp",
        model_card=".model QMPSA56 PNP(Is=12E-15 Bf=280 Vaf=110 Ikf=0.18)",
    ),
    # Darlington transistors
    BjtEntry(
        slug="tip120",
        model_name="QTIP120",
        description="NPN Darlington transistor 60V/5A",
        type="npn",
        model_card=".model QTIP120 NPN(Is=1E-13 Bf=10000 Vaf=200 Ikf=2)",
    ),
    BjtEntry(
        slug="tip125",
        model_name="QTIP125",
        description="PNP Darlington transistor -60V/-5A",
        type="pnp",
        model_card=".model QTIP125 PNP(Is=1E-13 Bf=8000 Vaf=180 Ikf=1.8)",
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
