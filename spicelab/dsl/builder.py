from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable as TypingIterable
from dataclasses import dataclass
from typing import TypeVar

from ..core.circuit import Circuit
from ..core.components import (
    CCCS,
    CCVS,
    VCCS,
    VCVS,
    Capacitor,
    Component,
    Idc,
    Inductor,
    ISwitch,
    Resistor,
    Vdc,
    VSwitch,
)
from ..core.net import GND, Net, Port


@dataclass
class NodeAlias:
    name: str
    net: Net

    def __repr__(self) -> str:  # pragma: no cover - repr helper
        return f"NodeAlias(name={self.name!r}, net={self.net!r})"


class CircuitBuilder:
    """Helper to assemble complex circuits with named nodes.

    The builder keeps track of created components and nets, automatically creating
    node aliases when referenced by name. It is designed for interactive use in
    notebooks, where concise code matters more than boilerplate.
    """

    def __init__(self, name: str, *, ground_aliases: TypingIterable[str] | None = None) -> None:
        self._circuit = Circuit(name)
        self._aliases: dict[str, Net] = {}
        self._component_counters: defaultdict[str, int] = defaultdict(int)
        self._alias_display_used: dict[Net, bool] = {}
        self._string_use_count: defaultdict[Net, int] = defaultdict(int)
        aliases = {"0", "gnd", "ground"}
        if ground_aliases:
            aliases.update(ground_aliases)
        for alias in aliases:
            self._aliases[alias.lower()] = GND

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    @property
    def circuit(self) -> Circuit:
        """Return the underlying :class:`spicelab.core.circuit.Circuit`."""

        return self._circuit

    def net(self, name: str, *, alias: str | None = None) -> Net:
        """Create or retrieve a named net.

        Parameters
        ----------
        name:
            The canonical name of the net.
        alias:
            Optional secondary name that will map to the same net.
        """

        net = self._aliases.get(name.lower())
        if net is None or net is GND and name != "0":
            net = Net(name)
            self._aliases[name.lower()] = net
        if alias:
            self._aliases[alias.lower()] = net
        return net

    # Component shorthands ------------------------------------------------
    def vdc(
        self, positive: str | Net, negative: str | Net, *, value: str, ref: str | None = None
    ) -> Vdc:
        return self._place_component(
            Vdc(ref=ref or self._next_ref("V"), value=value), positive, negative
        )

    def idc(
        self, positive: str | Net, negative: str | Net, *, value: str, ref: str | None = None
    ) -> Idc:
        return self._place_component(
            Idc(ref=ref or self._next_ref("I"), value=value), positive, negative
        )

    def resistor(
        self, a: str | Net, b: str | Net, *, value: str, ref: str | None = None
    ) -> Resistor:
        return self._place_component(
            Resistor(ref=ref or self._next_ref("R"), value=value),
            a,
            b,
        )

    def capacitor(
        self, a: str | Net, b: str | Net, *, value: str, ref: str | None = None
    ) -> Capacitor:
        return self._place_component(
            Capacitor(ref=ref or self._next_ref("C"), value=value),
            a,
            b,
        )

    def inductor(
        self, a: str | Net, b: str | Net, *, value: str, ref: str | None = None
    ) -> Inductor:
        return self._place_component(
            Inductor(ref=ref or self._next_ref("L"), value=value),
            a,
            b,
        )

    def vcvs(
        self,
        positive: str | Net,
        negative: str | Net,
        ctrl_positive: str | Net,
        ctrl_negative: str | Net,
        *,
        gain: str,
        ref: str | None = None,
    ) -> VCVS:
        return self._place_component(
            VCVS(ref=ref or self._next_ref("E"), gain=gain),
            positive,
            negative,
            ctrl_positive,
            ctrl_negative,
        )

    def vccs(
        self,
        positive: str | Net,
        negative: str | Net,
        ctrl_positive: str | Net,
        ctrl_negative: str | Net,
        *,
        gm: str,
        ref: str | None = None,
    ) -> VCCS:
        return self._place_component(
            VCCS(ref=ref or self._next_ref("G"), gm=gm),
            positive,
            negative,
            ctrl_positive,
            ctrl_negative,
        )

    def cccs(
        self,
        positive: str | Net,
        negative: str | Net,
        ctrl_vsrc: str,
        *,
        gain: str,
        ref: str | None = None,
    ) -> CCCS:
        comp = CCCS(ref=ref or self._next_ref("F"), ctrl_vsrc=ctrl_vsrc, gain=gain)
        return self._place_component(comp, positive, negative)

    def ccvs(
        self,
        positive: str | Net,
        negative: str | Net,
        ctrl_vsrc: str,
        *,
        r: str,
        ref: str | None = None,
    ) -> CCVS:
        comp = CCVS(ref=ref or self._next_ref("H"), ctrl_vsrc=ctrl_vsrc, r=r)
        return self._place_component(comp, positive, negative)

    def vswitch(
        self,
        positive: str | Net,
        negative: str | Net,
        ctrl_positive: str | Net,
        ctrl_negative: str | Net,
        *,
        model: str,
        ref: str | None = None,
    ) -> VSwitch:
        return self._place_component(
            VSwitch(ref=ref or self._next_ref("S"), model=model),
            positive,
            negative,
            ctrl_positive,
            ctrl_negative,
        )

    def iswitch(
        self,
        positive: str | Net,
        negative: str | Net,
        ctrl_vsrc: str,
        *,
        model: str,
        ref: str | None = None,
    ) -> ISwitch:
        comp = ISwitch(ref=ref or self._next_ref("W"), ctrl_vsrc=ctrl_vsrc, model=model)
        return self._place_component(comp, positive, negative)

    # Generic helpers -----------------------------------------------------
    TComp = TypeVar("TComp", bound=Component)

    def place(self, component: TComp, *nodes: str | Net) -> TComp:
        """Place an existing component instance and connect it to given nodes."""

        return self._place_component(component, *nodes)

    def alias(self, existing: str, *aliases: str) -> None:
        """Add additional aliases for a named net."""

        net = self._resolve_net(existing)
        for alias in aliases:
            self._aliases[alias.lower()] = net
            if net is not GND and alias:
                try:
                    object.__setattr__(net, "name", alias)
                except Exception:  # pragma: no cover - defensive fallback
                    pass
        self._alias_display_used.pop(net, None)

    def bus(self, prefix: str, count: int) -> list[Net]:
        """Create a list of related nets (useful for buses or ladder circuits)."""

        nets = []
        for idx in range(count):
            name = f"{prefix}{idx}"
            nets.append(self.net(name))
        return nets

    def build(self) -> Circuit:
        """Return the assembled circuit."""

        return self._circuit

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _next_ref(self, prefix: str) -> str:
        self._component_counters[prefix] += 1
        return f"{prefix}{self._component_counters[prefix]}"

    def _resolve_net(self, key: str | Net) -> Net:
        if isinstance(key, Net):
            return key
        if isinstance(key, Port):  # pragma: no cover - defensive
            raise TypeError("Ports are not accepted here; pass a net name or Net instance")
        normalized = key.lower()
        net = self._aliases.get(normalized)
        if net is None:
            net = self.net(key)
        return net

    def _place_component(self, component: TComp, *nodes: str | Net) -> TComp:
        required = len(component.ports)
        if len(nodes) != required:
            raise ValueError(
                f"Component {component.ref} expects {required} nodes, got {len(nodes)}"
            )

        self._circuit.add(component)
        for port, node in zip(component.ports, nodes, strict=True):
            net = self._resolve_net(node)
            label: str | None = None
            if isinstance(node, str):
                display_name = getattr(net, "name", None)
                if net is GND:
                    label = None
                elif not display_name:
                    label = node
                elif node == display_name:
                    # Alias typed explicitly: rely on net display name once
                    label = None
                else:
                    # Alternate between canonical string and display name so both appear
                    k = self._string_use_count[net]
                    label = node if (k % 2 == 0) else None
                    self._string_use_count[net] = k + 1
            elif isinstance(node, Net) and node.name:
                label = str(node.name)
            self._circuit.connect_with_label(port, net, label=label)
            if label:
                self._alias_display_used[net] = True
        return component


__all__ = ["CircuitBuilder", "NodeAlias"]
