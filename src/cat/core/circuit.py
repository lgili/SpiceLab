from __future__ import annotations

from dataclasses import dataclass, field

from ..utils.log import get_logger
from .components import Component
from .net import GND, Net, Port

log = get_logger("cat.core.circuit")


@dataclass
class Circuit:
    name: str
    _net_ids: dict[Net, int] = field(default_factory=dict, init=False)
    _port_to_net: dict[Port, Net] = field(default_factory=dict, init=False)
    _components: list[Component] = field(default_factory=list, init=False)
    _directives: list[str] = field(default_factory=list, init=False)

    def add(self, *comps: Component) -> Circuit:
        for c in comps:
            self._components.append(c)
        return self

    def add_directive(self, line: str) -> Circuit:
        """Append a raw SPICE directive (e.g., ".model ...", ".param ...")."""
        self._directives.append(line.rstrip("\n"))
        return self

    def connect(self, a: Port, b: Net | Port) -> Circuit:
        if isinstance(b, Port):
            # Port-Port connect: map both to a shared Net
            na = self._port_to_net.get(a)
            nb = self._port_to_net.get(b)
            if na and nb and (na is not nb):
                # merge: re-map all ports of nb to na
                for p, n in list(self._port_to_net.items()):
                    if n is nb:
                        self._port_to_net[p] = na
            else:
                self._port_to_net[a] = na or nb or Net()
                self._port_to_net[b] = self._port_to_net[a]
        else:
            # Port-Net
            self._port_to_net[a] = b
        return self

    def _assign_node_ids(self) -> None:
        self._net_ids.clear()
        # always ensure GND is 0
        self._net_ids[GND] = 0
        next_id = 1
        for n in set(self._port_to_net.values()):
            if n is GND:
                continue
            if n not in self._net_ids:
                self._net_ids[n] = next_id
                next_id += 1

    def _net_of(self, p: Port) -> str:
        n = self._port_to_net.get(p)
        if n is None:
            raise ValueError(f"Unconnected port: {p.owner.ref}.{p.name}")
        node_id = self._net_ids.get(n)
        if node_id is None:
            raise RuntimeError("Node IDs not assigned.")
        return "0" if n is GND else f"n{node_id}"

    def validate(self) -> None:
        # each component must have all ports connected
        for comp in self._components:
            for port in comp.ports:
                if port not in self._port_to_net:
                    raise ValueError(f"Unconnected port: {comp.ref}.{port.name}")

    def build_netlist(self) -> str:
        self.validate()
        self._assign_node_ids()
        lines = [f"* {self.name}"]
        for comp in self._components:
            lines.append(comp.spice_card(self._net_of))
        # Append directives (if any) before .end
        for d in self._directives:
            lines.append(d)
        lines.append(".end")
        return "\n".join(lines)
