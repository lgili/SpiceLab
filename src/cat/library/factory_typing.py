"""Typing helpers (Protocols) for component factories."""

from __future__ import annotations

from typing import Any, Protocol

from ..core.components import (
    CCCS,
    CCVS,
    VCCS,
    VCVS,
    Capacitor,
    Diode,
    Idc,
    Inductor,
    ISwitch,
    OpAmpIdeal,
    Resistor,
    VSwitch,
)
from .opamps import OpAmpSubckt
from .transistors import Bjt, Mosfet


class ComponentFactoryProtocol(Protocol):
    """Generic interface for factories returning any component."""

    def __call__(self, *args: Any, **kwargs: Any) -> object: ...


class ResistorFactory(Protocol):
    def __call__(self, ref: str, *, value: str | float | None = None) -> Resistor: ...


class CapacitorFactory(Protocol):
    def __call__(self, ref: str, *, value: str | float | None = None) -> Capacitor: ...


class InductorFactory(Protocol):
    def __call__(self, ref: str, *, value: str | float | None = None) -> Inductor: ...


class DiodeFactory(Protocol):
    def __call__(self, ref: str, *, model: str | None = None) -> Diode: ...


class CurrentSourceFactory(Protocol):
    def __call__(self, ref: str, *, value: str | float | None = None) -> Idc: ...


class VSwitchFactory(Protocol):
    def __call__(self, ref: str, *, model: str | None = None) -> VSwitch: ...


class ISwitchFactory(Protocol):
    def __call__(
        self,
        ref: str,
        *,
        ctrl_vsrc: str = "VCTRL",
        model: str | None = None,
    ) -> ISwitch: ...


class MosfetFactory(Protocol):
    def __call__(
        self,
        ref: str,
        *,
        model: str | None = None,
        params: str | None = None,
    ) -> Mosfet: ...


class BjtFactory(Protocol):
    def __call__(
        self,
        ref: str,
        *,
        model: str | None = None,
        area: str | float | None = None,
    ) -> Bjt: ...


class IdealOpAmpFactory(Protocol):
    def __call__(
        self,
        ref: str,
        *,
        gain: str | float | None = None,
    ) -> OpAmpIdeal: ...


class OpAmpSubcktFactory(Protocol):
    def __call__(self, ref: str, *, subckt_name: str | None = None) -> OpAmpSubckt: ...


class ControlledSourceFactory(Protocol):
    def __call__(self, ref: str, *, gain: str | float) -> VCVS | VCCS | CCCS | CCVS: ...


__all__ = [
    "ComponentFactoryProtocol",
    "ResistorFactory",
    "CapacitorFactory",
    "InductorFactory",
    "DiodeFactory",
    "CurrentSourceFactory",
    "VSwitchFactory",
    "ISwitchFactory",
    "MosfetFactory",
    "BjtFactory",
    "IdealOpAmpFactory",
    "OpAmpSubcktFactory",
    "ControlledSourceFactory",
]
