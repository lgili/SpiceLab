from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .net import Port, PortRole
from .typing import HasPorts


@dataclass
class Component(HasPorts):
    ref: str
    value: str | float
    _ports: tuple[Port, ...] = field(init=False, repr=False)

    @property
    def ports(self) -> tuple[Port, ...]:
        return self._ports

    def spice_card(self, net_of: Callable[[Port], str]) -> str:
        raise NotImplementedError


@dataclass
class Resistor(Component):
    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_ports",
            (Port(self, "a", PortRole.NODE), Port(self, "b", PortRole.NODE)),
        )

    def spice_card(self, net_of: Callable[[Port], str]) -> str:
        a, b = self.ports
        return f"R{self.ref} {net_of(a)} {net_of(b)} {self.value}"


@dataclass
class Capacitor(Component):
    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_ports",
            (Port(self, "a", PortRole.NODE), Port(self, "b", PortRole.NODE)),
        )

    def spice_card(self, net_of: Callable[[Port], str]) -> str:
        a, b = self.ports
        return f"C{self.ref} {net_of(a)} {net_of(b)} {self.value}"


@dataclass
class Vdc(Component):
    """DC voltage source; ports: p (positive), n (negative)."""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_ports",
            (Port(self, "p", PortRole.POSITIVE), Port(self, "n", PortRole.NEGATIVE)),
        )

    def spice_card(self, net_of: Callable[[Port], str]) -> str:
        p, n = self.ports
        return f"V{self.ref} {net_of(p)} {net_of(n)} DC {self.value}"


@dataclass
class Vac(Component):
    """AC small-signal voltage source for .AC analysis; ports: p (positive), n (negative).

    value: pode ser usado como label/descrição (ignoramos no card).
    ac_mag: magnitude AC (volts RMS ou conforme convenção do simulador).
    ac_phase: fase em graus (opcional).
    """

    ac_mag: float = 1.0
    ac_phase: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_ports",
            (Port(self, "p", PortRole.POSITIVE), Port(self, "n", PortRole.NEGATIVE)),
        )

    def spice_card(self, net_of: Callable[[Port], str]) -> str:
        p, n = self.ports
        # Para AC, tipicamente não precisamos de termo DC; usamos apenas "AC mag [phase]"
        if self.ac_phase:
            return f"V{self.ref} {net_of(p)} {net_of(n)} AC {self.ac_mag} {self.ac_phase}"
        return f"V{self.ref} {net_of(p)} {net_of(n)} AC {self.ac_mag}"


# Helpers auto-ref (amigáveis para notebooks)
_counter: dict[str, int] = {}


def _next(prefix: str) -> str:
    _counter[prefix] = _counter.get(prefix, 0) + 1
    return str(_counter[prefix])


def R(value: str | float) -> Resistor:
    return Resistor(ref=_next("R"), value=value)


def C(value: str | float) -> Capacitor:
    return Capacitor(ref=_next("C"), value=value)


def V(value: str | float) -> Vdc:
    return Vdc(ref=_next("V"), value=value)


def VA(ac_mag: float = 1.0, ac_phase: float = 0.0, label: str | float = "") -> Vac:
    # label é apenas para manter compatibilidade com assinatura parecida; não é usado no card
    return Vac(ref=_next("V"), value=str(label), ac_mag=ac_mag, ac_phase=ac_phase)
