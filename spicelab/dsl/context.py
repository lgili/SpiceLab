from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from dataclasses import dataclass

from ..core.circuit import Circuit as CoreCircuit
from ..core.components import Capacitor, Component, Inductor, Resistor, Vdc
from ..core.net import GND
from ..core.net import Net as CoreNet
from .expressions import normalize_expression

_NodeLike = str | CoreNet
_CURRENT_CONTEXT: ContextVar[DesignContext | None] = ContextVar(
    "spicelab_dsl_context", default=None
)

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DSLContextError(RuntimeError):
    """Raised when DSL helpers are used outside an active Circuit context."""


@dataclass
class DesignContext(AbstractContextManager["DesignContext"]):
    name: str
    ground_aliases: Iterable[str] | None = None

    def __post_init__(self) -> None:
        self._circuit = CoreCircuit(self.name)
        self._aliases: dict[str, CoreNet] = {}
        aliases = {"0", "gnd", "ground"}
        if self.ground_aliases:
            aliases.update(str(alias).lower() for alias in self.ground_aliases)
        for alias in aliases:
            self._aliases[alias] = GND
        self._token: Token[DesignContext | None] | None = None

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------
    def __enter__(self) -> DesignContext:
        current = _CURRENT_CONTEXT.get()
        if current is not None:
            raise DSLContextError("nested spicelab.dsl Circuit contexts are not supported")
        self._token = _CURRENT_CONTEXT.set(self)
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self._token is not None:
            _CURRENT_CONTEXT.reset(self._token)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    @property
    def circuit(self) -> CoreCircuit:
        return self._circuit

    # Net operations ----------------------------------------------------
    def net(self, name: str | None = None, *, alias: str | None = None) -> CoreNet:
        if name is None:
            net = CoreNet()
            return net
        net = self._get_or_create_net(name)
        if alias:
            self._aliases[str(alias).lower()] = net
        return net

    def resolve_node(self, node: _NodeLike) -> CoreNet:
        if isinstance(node, CoreNet):
            return node
        if isinstance(node, str):
            return self._get_or_create_net(node)
        raise TypeError(f"unsupported node type: {type(node)!r}")

    def add_component(self, component: Component, nodes: Sequence[_NodeLike]) -> Component:
        required = len(component.ports)
        if len(nodes) != required:
            raise ValueError(
                f"component {component.ref} expects {required} nodes, got {len(nodes)}"
            )
        self._circuit.add(component)
        for port, node in zip(component.ports, nodes, strict=True):
            net = self.resolve_node(node)
            self._circuit.connect(port, net)
        return component

    # Directive helpers -------------------------------------------------
    def add_param(self, name: str, value: str | float | int | bool) -> None:
        if not _NAME_RE.match(name):
            raise ValueError(f"invalid parameter name: {name!r}")
        expr = normalize_expression(value)
        self._circuit.add_directive(f".param {name}={expr}")

    def add_option(self, **options: str | float | int | bool) -> None:
        if not options:
            raise ValueError("Option() requires at least one key=value pair")
        parts = []
        for key, val in options.items():
            if not isinstance(key, str) or not key:
                raise ValueError("option names must be non-empty strings")
            parts.append(f"{key.lower()}={normalize_expression(val)}")
        self._circuit.add_directive(".option " + " ".join(parts))

    def add_temp(self, *values: str | float | int | bool) -> None:
        if not values:
            raise ValueError("TEMP() requires at least one temperature value")
        exprs = [normalize_expression(val) for val in values]
        self._circuit.add_directive(".temp " + " ".join(exprs))

    def add_ic(self, **assignments: str | float | int | bool) -> None:
        if not assignments:
            raise ValueError("IC() requires at least one node=value pair")
        parts: list[str] = []
        for key, value in assignments.items():
            if not isinstance(key, str) or not key:
                raise ValueError("IC keys must be non-empty strings")
            expr = normalize_expression(value)
            key_upper = key.strip().upper()
            if key_upper.startswith("V(") or key_upper.startswith("I("):
                spec = f"{key.strip()}={expr}"
            else:
                spec = f"V({key.strip()})={expr}"
            parts.append(spec)
        self._circuit.add_directive(".ic " + " ".join(parts))

    def add_directive(self, text: str, *, safe: bool = True) -> None:
        if not text:
            raise ValueError("Directive text cannot be empty")
        line = text.strip()
        if "\n" in line or "\r" in line:
            raise ValueError("Directive text cannot contain newlines")
        if safe and not line.startswith("."):
            raise ValueError("Safe directives must start with a '.'")
        self._circuit.add_directive(line)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_or_create_net(self, name: str) -> CoreNet:
        key = name.lower()
        net = self._aliases.get(key)
        if net is not None:
            return net
        net = GND if key in {"0", "gnd", "ground"} else CoreNet(name)
        self._aliases[key] = net
        if net is not GND and net.name is None:
            try:
                object.__setattr__(net, "name", name)
            except Exception:
                pass
        return net


def _require_context(action: str) -> DesignContext:
    ctx = _CURRENT_CONTEXT.get()
    if ctx is None:
        raise DSLContextError(f"{action} can only be used inside spicelab.dsl.Circuit context")
    return ctx


def Circuit(name: str, *, ground_aliases: Iterable[str] | None = None) -> DesignContext:
    return DesignContext(name=name, ground_aliases=ground_aliases)


def Net(name: str | None = None, *, alias: str | None = None) -> CoreNet:
    ctx = _require_context("Net()")
    return ctx.net(name, alias=alias)


def Param(name: str, value: str | float | int | bool) -> None:
    ctx = _require_context("Param()")
    ctx.add_param(name, value)


def Option(**options: str | float | int | bool) -> None:
    ctx = _require_context("Option()")
    ctx.add_option(**options)


def TEMP(*values: str | float | int | bool) -> None:
    ctx = _require_context("TEMP()")
    ctx.add_temp(*values)


def IC(**assignments: str | float | int | bool) -> None:
    ctx = _require_context("IC()")
    ctx.add_ic(**assignments)


def Directive(text: str, *, safe: bool = True) -> None:
    ctx = _require_context("Directive()")
    ctx.add_directive(text, safe=safe)


def place(component: Component, *nodes: _NodeLike) -> Component:
    ctx = _require_context("Component placement")
    return ctx.add_component(component, nodes)


def R(ref: str, a: _NodeLike, b: _NodeLike, value: str | float | int | bool) -> Resistor:
    result: Resistor = place(Resistor(ref=ref, value=normalize_expression(value)), a, b)  # type: ignore[assignment]
    return result


def C(ref: str, a: _NodeLike, b: _NodeLike, value: str | float | int | bool) -> Capacitor:
    result: Capacitor = place(Capacitor(ref=ref, value=normalize_expression(value)), a, b)  # type: ignore[assignment]
    return result


def L(ref: str, a: _NodeLike, b: _NodeLike, value: str | float | int | bool) -> Inductor:
    result: Inductor = place(Inductor(ref=ref, value=normalize_expression(value)), a, b)  # type: ignore[assignment]
    return result


def V(ref: str, positive: _NodeLike, negative: _NodeLike, value: str | float | int | bool) -> Vdc:
    result: Vdc = place(Vdc(ref=ref, value=normalize_expression(value)), positive, negative)  # type: ignore[assignment]
    return result


__all__ = [
    "Circuit",
    "DesignContext",
    "Directive",
    "DSLContextError",
    "IC",
    "L",
    "Net",
    "Option",
    "Param",
    "R",
    "TEMP",
    "V",
    "C",
    "place",
]
