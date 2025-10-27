from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ..core.circuit import Circuit
from ..core.components import Component
from .builder import CircuitBuilder

ComponentKind = Literal[
    "resistor",
    "capacitor",
    "inductor",
    "vdc",
    "idc",
    "vcvs",
    "vccs",
    "cccs",
    "ccvs",
    "vswitch",
    "iswitch",
]


class NetSpec(BaseModel):
    """Declarative description of a circuit net."""

    name: str
    aliases: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_aliases(self) -> NetSpec:
        lowered = [alias.lower() for alias in self.aliases]
        if len(lowered) != len(set(lowered)):
            raise ValueError("duplicate aliases detected for net")
        return self


class ComponentSpec(BaseModel):
    """Declarative component placement in the circuit DSL."""

    kind: ComponentKind
    nodes: list[str]
    value: str | float | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    ref: str | None = None

    def ensure_nodes(self, expected: int) -> None:
        if len(self.nodes) != expected:
            raise ValueError(
                f"component '{self.kind}' expects {expected} nodes, got {len(self.nodes)}"
            )

    def require_value(self, description: str) -> str | float:
        if self.value is None:
            raise ValueError(f"component '{self.kind}' requires a value for {description}")
        return self.value

    def require_param(self, name: str) -> Any:
        if name not in self.params:
            raise ValueError(f"component '{self.kind}' requires parameter '{name}'")
        return self.params[name]


class CircuitSpec(BaseModel):
    """High-level declarative description of a circuit."""

    name: str
    components: list[ComponentSpec]
    nets: list[NetSpec] = Field(default_factory=list)
    directives: list[str] = Field(default_factory=list)
    ground_aliases: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> CircuitSpec:
        names = [net.name.lower() for net in self.nets]
        if len(names) != len(set(names)):
            raise ValueError("duplicate net names are not allowed")
        return self


@dataclass
class BuildResult:
    circuit: Circuit
    components: dict[str, Component]


BuilderFn = Callable[[CircuitBuilder, ComponentSpec], Component]


def _build_resistor(builder: CircuitBuilder, spec: ComponentSpec) -> Component:
    spec.ensure_nodes(2)
    value = spec.require_value("resistance")
    return builder.resistor(spec.nodes[0], spec.nodes[1], value=str(value), ref=spec.ref)


def _build_capacitor(builder: CircuitBuilder, spec: ComponentSpec) -> Component:
    spec.ensure_nodes(2)
    value = spec.require_value("capacitance")
    return builder.capacitor(spec.nodes[0], spec.nodes[1], value=str(value), ref=spec.ref)


def _build_inductor(builder: CircuitBuilder, spec: ComponentSpec) -> Component:
    spec.ensure_nodes(2)
    value = spec.require_value("inductance")
    return builder.inductor(spec.nodes[0], spec.nodes[1], value=str(value), ref=spec.ref)


def _build_vdc(builder: CircuitBuilder, spec: ComponentSpec) -> Component:
    spec.ensure_nodes(2)
    value = spec.require_value("DC voltage")
    return builder.vdc(spec.nodes[0], spec.nodes[1], value=str(value), ref=spec.ref)


def _build_idc(builder: CircuitBuilder, spec: ComponentSpec) -> Component:
    spec.ensure_nodes(2)
    value = spec.require_value("DC current")
    return builder.idc(spec.nodes[0], spec.nodes[1], value=str(value), ref=spec.ref)


def _build_vcvs(builder: CircuitBuilder, spec: ComponentSpec) -> Component:
    spec.ensure_nodes(4)
    gain = spec.require_value("gain")
    return builder.vcvs(
        spec.nodes[0],
        spec.nodes[1],
        spec.nodes[2],
        spec.nodes[3],
        gain=str(gain),
        ref=spec.ref,
    )


def _build_vccs(builder: CircuitBuilder, spec: ComponentSpec) -> Component:
    spec.ensure_nodes(4)
    gm = spec.require_value("transconductance")
    return builder.vccs(
        spec.nodes[0],
        spec.nodes[1],
        spec.nodes[2],
        spec.nodes[3],
        gm=str(gm),
        ref=spec.ref,
    )


def _build_cccs(builder: CircuitBuilder, spec: ComponentSpec) -> Component:
    spec.ensure_nodes(2)
    ctrl_vsrc = str(spec.require_param("ctrl_vsrc"))
    gain = spec.require_value("current gain")
    return builder.cccs(
        spec.nodes[0],
        spec.nodes[1],
        ctrl_vsrc=ctrl_vsrc,
        gain=str(gain),
        ref=spec.ref,
    )


def _build_ccvs(builder: CircuitBuilder, spec: ComponentSpec) -> Component:
    spec.ensure_nodes(2)
    ctrl_vsrc = str(spec.require_param("ctrl_vsrc"))
    resistance = spec.require_value("transresistance")
    return builder.ccvs(
        spec.nodes[0],
        spec.nodes[1],
        ctrl_vsrc=ctrl_vsrc,
        r=str(resistance),
        ref=spec.ref,
    )


def _build_vswitch(builder: CircuitBuilder, spec: ComponentSpec) -> Component:
    spec.ensure_nodes(4)
    model = str(spec.require_param("model"))
    return builder.vswitch(
        spec.nodes[0],
        spec.nodes[1],
        spec.nodes[2],
        spec.nodes[3],
        model=model,
        ref=spec.ref,
    )


def _build_iswitch(builder: CircuitBuilder, spec: ComponentSpec) -> Component:
    spec.ensure_nodes(2)
    model = str(spec.require_param("model"))
    ctrl_vsrc = str(spec.require_param("ctrl_vsrc"))
    return builder.iswitch(
        spec.nodes[0],
        spec.nodes[1],
        ctrl_vsrc=ctrl_vsrc,
        model=model,
        ref=spec.ref,
    )


_HANDLERS: dict[ComponentKind, BuilderFn] = {
    "resistor": _build_resistor,
    "capacitor": _build_capacitor,
    "inductor": _build_inductor,
    "vdc": _build_vdc,
    "idc": _build_idc,
    "vcvs": _build_vcvs,
    "vccs": _build_vccs,
    "cccs": _build_cccs,
    "ccvs": _build_ccvs,
    "vswitch": _build_vswitch,
    "iswitch": _build_iswitch,
}


def build_circuit_from_spec(spec: CircuitSpec) -> BuildResult:
    """Materialize a :class:`Circuit` from a declarative specification."""

    builder = CircuitBuilder(spec.name, ground_aliases=spec.ground_aliases)

    for net_spec in spec.nets:
        net_obj = builder.net(net_spec.name)
        for alias in net_spec.aliases:
            builder.alias(net_spec.name, alias)
        if net_obj.name is None:
            try:
                object.__setattr__(net_obj, "name", net_spec.name)
            except Exception:  # pragma: no cover - safety guard
                pass

    components: dict[str, Component] = {}
    for comp_spec in spec.components:
        handler = _HANDLERS.get(comp_spec.kind)
        if handler is None:
            raise ValueError(f"unsupported component kind: {comp_spec.kind!r}")
        component = handler(builder, comp_spec)
        ref = getattr(component, "ref", None)
        if isinstance(ref, str):
            components[ref] = component

    circuit = builder.build()
    for directive in spec.directives:
        circuit.add_directive(directive)

    return BuildResult(circuit=circuit, components=components)


def load_circuit_spec(data: dict[str, Any] | CircuitSpec) -> BuildResult:
    """Helper that accepts raw dicts (e.g. JSON payloads)."""

    spec = data if isinstance(data, CircuitSpec) else CircuitSpec.model_validate(data)
    return build_circuit_from_spec(spec)


__all__ = [
    "BuildResult",
    "CircuitSpec",
    "ComponentSpec",
    "NetSpec",
    "build_circuit_from_spec",
    "load_circuit_spec",
]
