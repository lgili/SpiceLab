"""Shared-library adapter for ngspice co-simulation.

This module exposes a thin abstraction around *libngspice* so that higher-level
code can drive transient analyses using callbacks (``on_tran_point``) and
provide external sources computed in Python (e.g. controllers or ADC models).

The default implementation deliberately defers to an injected backend because
packaging ``libngspice`` across platforms is outside the scope of the toolkit.
Production code is expected to supply a backend that actually bridges to the
shared library (e.g. via :mod:`ctypes`).  The included tests exercise this layer
with a pure-Python fake backend, demonstrating the contract that real
implementations must follow.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = [
    "NgSpiceSharedAdapter",
    "SharedAdapterResult",
    "SharedCallbacks",
    "SharedTranPoint",
    "ExternalSourceRequest",
]


@dataclass(frozen=True)
class SharedTranPoint:
    """Information passed to ``on_tran_point`` callbacks."""

    time: float
    values: Mapping[str, float]


@dataclass(frozen=True)
class ExternalSourceRequest:
    """Context passed to external-source providers."""

    name: str
    time: float
    values: Mapping[str, float]


@dataclass
class SharedCallbacks:
    """Callback collection understood by the shared adapter."""

    on_init: Callable[[], None] | None = None
    on_exit: Callable[[int], None] | None = None
    on_message: Callable[[str], None] | None = None
    on_tran_point: Callable[[SharedTranPoint], None] | None = None


@dataclass
class SharedAdapterResult:
    """Outcome returned by :class:`NgSpiceSharedAdapter` backends."""

    dataset: Any
    log_lines: Sequence[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""


class SharedBackend(Protocol):
    """Protocol implemented by concrete libngspice backends."""

    def run(
        self,
        netlist: str,
        directives: Sequence[str],
        callbacks: SharedCallbacks | None,
        external_sources: Mapping[str, Callable[[ExternalSourceRequest], float]] | None,
    ) -> SharedAdapterResult: ...


def _load_default_backend() -> SharedBackend:
    from .ngspice_shared_backend import load_default_backend

    return load_default_backend()


class NgSpiceSharedAdapter:
    """High-level façade around a libngspice shared-library backend."""

    def __init__(self, backend: SharedBackend | None = None) -> None:
        self._backend = backend

    def run(
        self,
        netlist: str,
        directives: Sequence[str],
        callbacks: SharedCallbacks | None = None,
        external_sources: Mapping[str, Callable[[ExternalSourceRequest], float]] | None = None,
    ) -> SharedAdapterResult:
        backend = self._backend or _load_default_backend()
        self._backend = backend
        return backend.run(netlist, directives, callbacks, external_sources or {})
