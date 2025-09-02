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
