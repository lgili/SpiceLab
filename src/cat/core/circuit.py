from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
        # always ensure any logical GND (name == "0") maps to node id 0
        # include the global GND sentinel for convenience
        self._net_ids[GND] = 0
        next_id = 1
        for n in set(self._port_to_net.values()):
            # treat any net named "0" as ground, not only the sentinel object
            if (n is GND) or (getattr(n, "name", None) == "0"):
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
        # Preserve user-provided names for nets (except GND)
        # treat any net named "0" as ground
        if (n is GND) or (n.name == "0"):
            return "0"
        if n.name and n.name != "0":
            return n.name
        # auto-generate a name that avoids colliding with common user names (like 'n1')
        return f"net_{node_id}"

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

    # -----------------------------
    # Introspection / Preview
    # -----------------------------
    def summary(self) -> str:
        """Return a human-friendly connectivity summary and basic lint warnings."""
        # Map nets to connected ports (ref.port)
        net_to_ports: dict[str, list[str]] = {}
        unconnected: list[str] = []
        # Ensure we have stable net names even if build wasn't called
        # We'll map by object identity with temporary IDs
        temp_ids: dict[Any, int] = {}
        next_id = 1

        def name_of_net(n: Net) -> str:
            nonlocal next_id
            if n is GND:
                return "0"
            if n.name and n.name != "0":
                return n.name
            if n not in temp_ids:
                temp_ids[n] = next_id
                next_id += 1
            return f"n{temp_ids[n]}"

        for comp in self._components:
            for port in comp.ports:
                key = f"{comp.ref}.{port.name}"
                n = self._port_to_net.get(port)
                if n is None:
                    unconnected.append(key)
                else:
                    nn = name_of_net(n)
                    net_to_ports.setdefault(nn, []).append(key)

        lines: list[str] = [f"Circuit: {self.name}"]
        lines.append("Connections:")
        for nn, plist in sorted(net_to_ports.items()):
            plist_sorted = ", ".join(sorted(plist))
            lines.append(f"  {nn}: {plist_sorted}")

        # Lint
        warnings: list[str] = []
        for nn, plist in net_to_ports.items():
            if nn != "0" and len(plist) <= 1:
                warnings.append(f"Net '{nn}' has degree {len(plist)} (possible floating)")
        if unconnected:
            warnings.append(f"Unconnected ports: {', '.join(sorted(unconnected))}")
        if warnings:
            lines.append("Warnings:")
            for w in warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)

    def print_connectivity(self) -> None:
        print(self.summary())

    def to_dot(self) -> str:
        """Export a Graphviz DOT representation (components as boxes, nets as ellipses)."""
        # Build stable names
        self._assign_node_ids()
        comp_ids: dict[Component, str] = {}
        net_ids: dict[Net, str] = {}

        def comp_id(c: Component) -> str:
            if c not in comp_ids:
                comp_ids[c] = f"comp_{c.ref}"
            return comp_ids[c]

        def net_id(n: Net) -> str:
            if n not in net_ids:
                if n is GND:
                    net_ids[n] = "net_0"
                else:
                    name = n.name or f"n{self._net_ids.get(n, 0)}"
                    safe = name.replace('"', "'")
                    net_ids[n] = f"net_{safe}"
            return net_ids[n]

        lines: list[str] = []
        lines.append("graph circuit {\n  rankdir=LR;\n  node [fontname=Helvetica];\n}")
        out: list[str] = ["graph circuit {", "  rankdir=LR;"]
        # Net nodes
        seen_nets: set[Net] = set()
        for n in set(self._port_to_net.values()):
            nid = net_id(n)
            label = "0" if n is GND else (n.name or f"n{self._net_ids.get(n, 0)}")
            if n is GND:
                out.append(
                    f'  {nid} [shape=ellipse, label="{label}", style=filled, fillcolor=lightgray];'
                )
            else:
                out.append(f'  {nid} [shape=ellipse, label="{label}"];')
            seen_nets.add(n)
        # Component nodes
        for c in self._components:
            cid = comp_id(c)
            val = str(c.value)
            out.append(f'  {cid} [shape=box, label="{type(c).__name__} {c.ref}\\n{val}"];')
        # Edges
        for c in self._components:
            cid = comp_id(c)
            for port in c.ports:
                nn = self._port_to_net.get(port)
                if nn is None:
                    continue
                nid = net_id(nn)
                out.append(f'  {cid} -- {nid} [label="{port.name}"];')
        out.append("}")
        return "\n".join(out)

    def render_svg(self, out_path: str) -> bool:
        """Render DOT to SVG using the 'dot' executable (Graphviz). Returns True if saved."""
        dot = shutil.which("dot")
        if not dot:
            return False
        tmp_dot = Path(out_path).with_suffix(".dot")
        tmp_dot.write_text(self.to_dot(), encoding="utf-8")
        proc = subprocess.run([dot, "-Tsvg", str(tmp_dot), "-o", out_path], capture_output=True)
        return proc.returncode == 0
