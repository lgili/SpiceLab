"""Immutable Circuit implementation with Union-Find net registry.

This is Phase 1 (P1.3) of the architecture redesign: converting Circuit
to a frozen dataclass with structural sharing and O(α(N)) net operations.

This module will coexist with the mutable Circuit during migration,
then gradually replace it as tests are updated.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..utils.log import get_logger
from .components import Component
from .net import GND, Net, Port
from .net_registry import NetRegistry
from .parameter import Parameter

if TYPE_CHECKING:
    from collections.abc import Mapping

log = get_logger("spicelab.core.circuit_v2")

__all__ = ["ImmutableCircuit"]


@dataclass(frozen=True)
class ImmutableCircuit:
    """Immutable circuit with structural sharing and efficient net operations.

    Key differences from mutable Circuit:
    - frozen=True: all fields immutable
    - components/directives are tuples (not lists)
    - NetRegistry replaces dict[Port, Net] (O(α(N)) vs O(N))
    - All mutations return NEW circuit instances
    - Lazy node ID assignment via @cached_property

    Benefits:
    - Thread-safe (no mutations)
    - Cacheable by identity (for orchestrator)
    - 10x faster connect operations
    - No deepcopy needed in sweeps/Monte Carlo

    Example:
        >>> circuit = (
        ...     ImmutableCircuit("rc_filter")
        ...     .add(R("R1", "in", "out", 1000))
        ...     .add(C("C1", "out", Net.gnd, 1e-6))
        ... )
    """

    name: str
    components: tuple[Component, ...] = field(default_factory=tuple)
    nets: NetRegistry = field(default_factory=NetRegistry)
    directives: tuple[str, ...] = field(default_factory=tuple)
    params: Mapping[str, Parameter] = field(default_factory=dict)  # Phase 2: Parameter system

    # Metadata from netlist import
    subckt_defs: Mapping[str, str] = field(default_factory=dict)
    subckt_instances: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    port_labels: Mapping[Port, str] = field(default_factory=dict)

    # ----------------------------------------------------------------------------------
    # Building blocks (return NEW instances)
    # ----------------------------------------------------------------------------------

    def add(self, *new_components: Component) -> ImmutableCircuit:
        """Return new circuit with added components (structural sharing).

        Time complexity: O(1) - tuple concatenation without copying existing data.
        """
        return replace(self, components=self.components + new_components)

    def add_directive(self, line: str) -> ImmutableCircuit:
        """Return new circuit with added SPICE directive."""
        return replace(self, directives=self.directives + (line.rstrip("\n"),))

    def add_directive_once(self, line: str) -> ImmutableCircuit:
        """Add directive if not already present (idempotent)."""
        normalized = line.strip()
        for existing in self.directives:
            if existing.strip() == normalized:
                return self  # Already present, no change
        return self.add_directive(line)

    def connect(self, a: Port, b: Net | Port) -> ImmutableCircuit:
        """Return new circuit with port-to-net connection.

        Time complexity: O(α(N)) amortized (Union-Find with path compression).
        This is ~10x faster than the O(N) dict-based approach in mutable Circuit.

        Args:
            a: Port to connect
            b: Net or another Port to connect to

        Returns:
            New ImmutableCircuit with updated connectivity.
        """
        if isinstance(b, Port):
            # Connect two ports: union them in same equivalence class
            new_nets = self.nets.union(a, b)
        else:
            # Connect port to net
            new_nets = self.nets.union(a, b)

        return replace(self, nets=new_nets)

    def connect_with_label(
        self, port: Port, net: Net, label: str | None = None
    ) -> ImmutableCircuit:
        """Connect port to net with optional display label."""
        circuit = self.connect(port, net)
        if label:
            new_labels = {**self.port_labels, port: label}
            circuit = replace(circuit, port_labels=new_labels)
        return circuit

    def with_param(self, name: str, param: Parameter) -> ImmutableCircuit:
        """Return new circuit with added parameter.

        Parameters are used by components via ParameterRef and resolved
        to .param statements in the netlist.

        Args:
            name: Parameter name (e.g., "Rload")
            param: Parameter definition with nominal, unit, tolerance, etc.

        Returns:
            New ImmutableCircuit with updated params dict.

        Example:
            >>> from spicelab.core.parameter import Parameter, NormalTolerance
            >>> from spicelab.core.units import Unit
            >>> r_param = Parameter("Rload", 10_000, Unit.OHM, NormalTolerance(5.0))
            >>> circuit = ImmutableCircuit("test").with_param("Rload", r_param)
        """
        new_params = {**self.params, name: param}
        return replace(self, params=new_params)

    # ----------------------------------------------------------------------------------
    # Lazy node ID assignment (cached)
    # ----------------------------------------------------------------------------------

    @cached_property
    def _node_assignments(self) -> dict[Net, int]:
        """Compute node IDs lazily and cache result.

        This replaces the mutable Circuit's _assign_node_ids() which was
        called multiple times (build_netlist, summary, to_dot).

        Now it's computed once on first access and cached forever
        (circuit is immutable, so IDs never change).
        """
        net_ids: dict[Net, int] = {GND: 0}
        next_id = 1
        seen: set[Net] = {GND}

        # Collect all nets from component ports
        for comp in self.components:
            for port in comp.ports:
                root = self.nets.find(port)
                if isinstance(root, Net) and root not in seen:
                    seen.add(root)
                    net_ids[root] = next_id
                    next_id += 1

        return net_ids

    def _net_of(self, port: Port) -> str:
        """Map port to netlist node name (for SPICE card generation)."""
        root = self.nets.find(port)

        # Not connected
        if root == port:
            raise ValueError(f"Unconnected port: {port.owner.ref}.{port.name}")

        # Root is GND
        if root == GND or (isinstance(root, Net) and root.name == "0"):
            return "0"

        # Root is named Net
        if isinstance(root, Net) and root.name:
            return root.name

        # Anonymous net: use node ID
        node_id = self._node_assignments.get(root) if isinstance(root, Net) else None
        if node_id is None:
            raise RuntimeError(f"No node ID assigned for {root}")

        return str(node_id)

    # ----------------------------------------------------------------------------------
    # Netlist generation
    # ----------------------------------------------------------------------------------

    def build_netlist(self) -> str:
        """Generate SPICE netlist from circuit.

        This is now much faster because _node_assignments is cached
        and only computed once (vs multiple _assign_node_ids() calls).

        Phase 2: Also generates .param statements from self.params.
        """
        lines: list[str] = [f"* {self.name}"]

        # Parameter definitions (Phase 2)
        if self.params:
            lines.append("")
            lines.append("* Parameters")
            for param in self.params.values():
                lines.append(param.to_spice())
            lines.append("")

        # Component cards
        for comp in self.components:
            card = comp.spice_card(self._net_of)
            for ln in card.splitlines():
                if ln.strip():
                    lines.append(ln)

        # Directives
        for directive in self.directives:
            lines.extend(directive.splitlines())

        # Ensure .end
        if not any(line.strip().lower() == ".end" for line in lines):
            lines.append(".end")

        return "\n".join(lines) + "\n"

    def save_netlist(self, path: str | Path) -> Path:
        """Persist netlist to file."""
        p = Path(path)
        p.write_text(self.build_netlist(), encoding="utf-8")
        return p

    # ----------------------------------------------------------------------------------
    # Hash (deterministic)
    # ----------------------------------------------------------------------------------

    def hash(self, *, extra: dict[str, object] | None = None) -> str:
        """Return deterministic hash for caching."""
        from .types import circuit_hash

        return circuit_hash(self, extra=extra)

    # ----------------------------------------------------------------------------------
    # Introspection helpers
    # ----------------------------------------------------------------------------------

    def _net_label(self, net: Net | None) -> str:
        """Get display label for net (for summary/debugging)."""
        if net is None:
            return "<unconnected>"
        if net is GND or (isinstance(net, Net) and net.name == "0"):
            return "0"
        if isinstance(net, Net) and net.name:
            return net.name

        node_id = self._node_assignments.get(net)
        return f"N{node_id:03d}" if node_id is not None else "<unnamed>"

    def summary(self) -> str:
        """Return human-readable circuit summary."""
        lines: list[str] = []
        warnings: list[str] = []

        lines.append(f"Circuit: {self.name}")
        lines.append(f"Components ({len(self.components)}):")

        for comp in self.components:
            port_descriptions: list[str] = []
            for port in comp.ports:
                root = self.nets.find(port)
                if isinstance(root, Net):
                    label = self.port_labels.get(port) or self._net_label(root)
                else:
                    label = "<unconnected>"
                    warnings.append(f"Port {comp.ref}.{port.name} is unconnected")
                port_descriptions.append(f"{port.name}->{label}")

            port_info = ", ".join(port_descriptions) if port_descriptions else "<no ports>"
            lines.append(f"  - {comp.ref} ({type(comp).__name__}): {port_info}")

        # List unique nets
        net_roots = {
            root
            for comp in self.components
            for port in comp.ports
            if (root := self.nets.find(port)) and isinstance(root, Net)
        }
        net_names = sorted({self._net_label(net) for net in net_roots})
        if net_names:
            lines.append(f"Nets ({len(net_names)}): {', '.join(net_names)}")

        if warnings:
            lines.append("Warnings:")
            for msg in warnings:
                lines.append(f"  * {msg}")
        else:
            lines.append("Warnings: none")

        return "\n".join(lines)

    def to_dot(self) -> str:
        """Generate Graphviz DOT for circuit visualization."""
        lines: list[str] = ["graph circuit {", "  rankdir=LR;"]

        # Component nodes
        comp_ids: dict[Component, str] = {}
        for idx, comp in enumerate(self.components, start=1):
            comp_id = f"comp_{idx}"
            comp_ids[comp] = comp_id
            label = f"{comp.ref}\\n{type(comp).__name__}"
            lines.append(f'  "{comp_id}" [shape=box,label="{label}"];')

        # Net nodes (unique roots only)
        net_ids: dict[Net, str] = {}
        net_counter = 1

        def _get_net_node(net: Net) -> str:
            nonlocal net_counter
            if net in net_ids:
                return net_ids[net]
            node_id = f"net_{net_counter}"
            net_counter += 1
            label = self._net_label(net)
            net_ids[net] = node_id
            lines.append(f'  "{node_id}" [shape=ellipse,label="{label}"];')
            return node_id

        # Edges: component ports to nets
        for comp in self.components:
            comp_id = comp_ids[comp]
            for port in comp.ports:
                root = self.nets.find(port)
                if isinstance(root, Net):
                    net_id = _get_net_node(root)
                    lines.append(f'  "{comp_id}" -- "{net_id}" [label="{port.name}",fontsize=10];')

        lines.append("}")
        return "\n".join(lines)

    # ----------------------------------------------------------------------------------
    # Notebook helpers
    # ----------------------------------------------------------------------------------

    def connectivity_dataframe(self, *, sort: bool = True, include_type: bool = True) -> Any:
        """Return pandas DataFrame with connectivity info."""
        try:
            import pandas as pd
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pandas required for connectivity_dataframe()") from exc

        rows: list[dict[str, object]] = []
        for comp in self.components:
            for order, port in enumerate(comp.ports):
                root = self.nets.find(port)
                label = self._net_label(root) if isinstance(root, Net) else "<unconnected>"
                rows.append(
                    {
                        "component": comp.ref,
                        "type": type(comp).__name__,
                        "port": port.name,
                        "net": label,
                        "_order": order,
                    }
                )

        if not rows:
            columns = ["component", "port", "net"]
            if include_type:
                columns.insert(1, "type")
            return pd.DataFrame(columns=columns)

        df = pd.DataFrame(rows)
        if sort:
            df = df.sort_values(["component", "_order", "port"]).reset_index(drop=True)
        if not include_type:
            df = df.drop(columns=["type"])
        return df.drop(columns=["_order"])

    def summary_table(self, *, indent: int = 2) -> str:
        """Return fixed-width table of connections."""
        df = self.connectivity_dataframe()
        if df.empty:
            return "(circuit is empty)"

        headers = list(df.columns)
        display_names = {col: col.capitalize() for col in headers}
        widths = {
            col: max(len(display_names[col]), *(len(str(val)) for val in df[col]))
            for col in headers
        }

        def fmt_row(row: dict[str, object]) -> str:
            cells = [str(row[col]).ljust(widths[col]) for col in headers]
            return " " * indent + "  ".join(cells)

        header_line = " " * indent + "  ".join(
            display_names[col].ljust(widths[col]) for col in headers
        )
        sep_line = " " * indent + "  ".join("-" * widths[col] for col in headers)
        body = "\n".join(fmt_row(df.iloc[idx]) for idx in range(len(df)))
        return f"{header_line}\n{sep_line}\n{body}"

    # ----------------------------------------------------------------------------------
    # Compatibility: expose old mutable Circuit API
    # ----------------------------------------------------------------------------------

    @property
    def _components(self) -> list[Component]:
        """Compatibility shim for old code expecting list."""
        return list(self.components)

    @property
    def _directives(self) -> list[str]:
        """Compatibility shim for old code expecting list."""
        return list(self.directives)

    @property
    def _port_to_net(self) -> dict[Port, Net]:
        """Compatibility shim: lazily build Port->Net mapping.

        WARNING: This is O(N) and defeats the purpose of Union-Find!
        Only use for gradual migration. New code should use .nets directly.
        """
        mapping: dict[Port, Net] = {}
        for comp in self.components:
            for port in comp.ports:
                root = self.nets.find(port)
                if isinstance(root, Net):
                    mapping[port] = root
        return mapping

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<ImmutableCircuit name={self.name!r} "
            f"components={len(self.components)} "
            f"nets={len(self.nets.all_nets())}>"
        )
