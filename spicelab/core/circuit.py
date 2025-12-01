# mypy: ignore-errors
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from ..utils.log import get_logger
from ..validators.circuit_validation import ValidationResult
from .components import (
    CCCS,
    CCVS,
    VCCS,
    VCVS,
    Capacitor,
    Component,
    Diode,
    Idc,
    Inductor,
    ISwitch,
    Resistor,
    Vdc,
    VSwitch,
)
from .net import GND, Net, Port
from .union_find import UnionFind

log = get_logger("spicelab.core.circuit")


@dataclass
class Circuit:
    """Logical circuit composed of components, nets and raw SPICE directives."""

    name: str
    _net_ids: dict[Net, int] = field(default_factory=dict, init=False)
    _port_to_net: dict[Port, Net] = field(default_factory=dict, init=False)
    _components: list[Component] = field(default_factory=list, init=False)
    _directives: list[str] = field(default_factory=list, init=False)
    # metadata captured when loading from existing netlists
    _subckt_defs: dict[str, str] = field(default_factory=dict, init=False)
    _subckt_instances: list[dict[str, object]] = field(default_factory=list, init=False)
    _port_labels: dict[Port, str] = field(default_factory=dict, init=False)
    # Union-Find for O(α(n)) net merging (M2 performance optimization)
    _net_union: UnionFind[Net] = field(default_factory=UnionFind, init=False)
    # Cache invalidation version counter
    _cache_version: int = field(default=0, init=False)

    # ----------------------------------------------------------------------------------
    # Building blocks
    # ----------------------------------------------------------------------------------
    def add(self, *components: Component) -> Circuit:
        """Append one or more components to the circuit."""

        self._components.extend(components)
        return self

    def add_directive(self, line: str) -> Circuit:
        """Append a raw SPICE directive (``.model``, ``.param`` ...)."""

        self._directives.append(line.rstrip("\n"))
        return self

    def add_directive_once(self, line: str) -> Circuit:
        """Add a directive if an identical line (ignoring whitespace) is absent."""

        normalized = line.strip()
        for existing in self._directives:
            if existing.strip() == normalized:
                return self
        return self.add_directive(line)

    def connect(self, a: Port, b: Net | Port) -> Circuit:
        """Connect a port to another port or to a logical net.

        Uses Union-Find for O(α(n)) amortized net merging (M2 optimization).
        """
        self._invalidate_cache()

        if isinstance(b, Port):
            net_a = self._port_to_net.get(a)
            net_b = self._port_to_net.get(b)

            if net_a and net_b and net_a is not net_b:
                # Merge using Union-Find: O(α(n)) instead of O(n)
                # Ensure both nets are in union-find
                if net_a not in self._net_union:
                    is_named_a = getattr(net_a, "name", None) is not None
                    self._net_union.make_set(net_a, net_a if is_named_a else None)
                if net_b not in self._net_union:
                    is_named_b = getattr(net_b, "name", None) is not None
                    self._net_union.make_set(net_b, net_b if is_named_b else None)

                # Prefer named net as canonical
                prefer = None
                if getattr(net_a, "name", None):
                    prefer = net_a
                elif getattr(net_b, "name", None):
                    prefer = net_b

                self._net_union.union(net_a, net_b, prefer=prefer)
            else:
                shared = net_a or net_b or Net()
                self._port_to_net[a] = shared
                self._port_to_net[b] = shared
                # Register in union-find
                if shared not in self._net_union:
                    is_named = getattr(shared, "name", None) is not None
                    self._net_union.make_set(shared, shared if is_named else None)
            self._port_labels.pop(b, None)
        else:
            self._port_to_net[a] = b
            # Register named net in union-find
            if b not in self._net_union:
                is_named = getattr(b, "name", None) is not None
                self._net_union.make_set(b, b if is_named else None)
        self._port_labels.pop(a, None)
        return self

    def _invalidate_cache(self) -> None:
        """Invalidate cached properties when circuit is modified."""
        self._cache_version += 1
        # Clear cached net IDs
        self._net_ids.clear()

    def connect_with_label(self, port: Port, net: Net, label: str | None = None) -> Circuit:
        """Connect ``port`` to ``net`` while recording a display label."""

        self.connect(port, net)
        if label:
            self._port_labels[port] = label
        return self

    # ----------------------------------------------------------------------------------
    # Net handling
    # ----------------------------------------------------------------------------------
    def _assign_node_ids(self) -> None:
        """Assign node IDs, using Union-Find for canonical net resolution."""
        self._net_ids.clear()
        self._net_ids[GND] = 0

        next_id = 1
        seen: set[Net] = {GND}

        def canonical_nets_from_components() -> Iterable[Net]:
            for comp in self._components:
                for port in comp.ports:
                    net = self._port_to_net.get(port)
                    if net is not None:
                        # Use canonical net for proper merging
                        yield self._get_canonical_net(net)

        for net in canonical_nets_from_components():
            if net in seen:
                continue
            seen.add(net)
            if getattr(net, "name", None) and net.name != "0":
                # preserve named nets but still assign an id for bookkeeping
                self._net_ids[net] = next_id
            else:
                self._net_ids[net] = next_id
            next_id += 1

    def _net_of(self, port: Port) -> str:
        net = self._port_to_net.get(port)
        if net is None:
            raise ValueError(f"Unconnected port: {port.owner.ref}.{port.name}")

        # Use Union-Find to get the canonical net (handles merged nets)
        canonical_net = self._get_canonical_net(net)

        if canonical_net is GND or getattr(canonical_net, "name", None) == "0":
            return "0"

        if getattr(canonical_net, "name", None):
            return str(canonical_net.name)

        node_id = self._net_ids.get(canonical_net)
        if node_id is None:
            raise RuntimeError("Node IDs not assigned")
        return str(node_id)

    def _get_canonical_net(self, net: Net) -> Net:
        """Get the canonical net for a possibly-merged net using Union-Find."""
        if net not in self._net_union:
            return net
        return self._net_union.get_canonical(net)

    # ----------------------------------------------------------------------------------
    # Netlist helpers
    # ----------------------------------------------------------------------------------
    def build_netlist(self) -> str:
        """Return a SPICE netlist representation of this circuit."""

        self._assign_node_ids()

        lines: list[str] = [f"* {self.name}"]

        for comp in self._components:
            card = comp.spice_card(self._net_of)
            # components such as AnalogMux may emit multi-line cards
            for ln in card.splitlines():
                if ln.strip():
                    lines.append(ln)

        for directive in self._directives:
            lines.extend(directive.splitlines())

        if not any(line.strip().lower() == ".end" for line in lines):
            lines.append(".end")

        return "\n".join(lines) + "\n"

    def preview_netlist(
        self,
        engine: str = "ngspice",
        *,
        syntax_highlight: bool = True,
        line_numbers: bool = True,
        show_stats: bool = True,
    ) -> str:
        """Preview the SPICE netlist with optional formatting.

        Generates the netlist and formats it for easy reading with optional
        syntax highlighting, line numbers, and circuit statistics.

        Args:
            engine: Target SPICE engine ("ngspice", "ltspice", "xyce").
                    Currently affects header comment only.
            syntax_highlight: Add ANSI colors for terminal display (default True)
            line_numbers: Add line numbers (default True)
            show_stats: Show circuit statistics header (default True)

        Returns:
            Formatted netlist string

        Example:
            >>> print(circuit.preview_netlist())
            ═══ Circuit: my_amp ═══
            Components: 5 | Nets: 4 | Directives: 1
            ───────────────────────
               1 │ * my_amp
               2 │ R1 vin vout 10k
               3 │ C1 vout 0 100n
               4 │ .end

            >>> # Without formatting
            >>> print(circuit.preview_netlist(syntax_highlight=False, line_numbers=False))
        """
        netlist = self.build_netlist()
        lines = netlist.rstrip().split("\n")

        # ANSI color codes
        RESET = "\033[0m"
        COMMENT = "\033[90m"  # Gray
        DIRECTIVE = "\033[33m"  # Yellow
        COMPONENT = "\033[36m"  # Cyan
        VALUE = "\033[32m"  # Green
        HEADER = "\033[1;34m"  # Bold blue

        def colorize_line(line: str) -> str:
            """Apply syntax highlighting to a netlist line."""
            if not syntax_highlight:
                return line

            stripped = line.strip()
            if not stripped:
                return line

            # Comments
            if stripped.startswith("*"):
                return f"{COMMENT}{line}{RESET}"

            # Directives (.model, .param, .end, etc.)
            if stripped.startswith("."):
                return f"{DIRECTIVE}{line}{RESET}"

            # Component lines (R1, C1, V1, etc.)
            parts = stripped.split()
            if parts and parts[0] and parts[0][0].upper() in "RCLVIDEQMJKXBSGWFH":
                # Highlight component ref and value
                if len(parts) >= 1:
                    ref = parts[0]
                    rest = " ".join(parts[1:])
                    # Try to highlight the last part as value
                    if len(parts) > 2:
                        nodes = " ".join(parts[1:-1])
                        value = parts[-1]
                        return f"{COMPONENT}{ref}{RESET} {nodes} {VALUE}{value}{RESET}"
                    return f"{COMPONENT}{ref}{RESET} {rest}"

            return line

        output_lines: list[str] = []

        # Statistics header
        if show_stats:
            self._assign_node_ids()
            n_components = len(self._components)
            n_nets = len(set(self._net_ids.values()))
            n_directives = len(self._directives)

            if syntax_highlight:
                output_lines.append(f"{HEADER}═══ Circuit: {self.name} ({engine}) ═══{RESET}")
                output_lines.append(
                    f"Components: {n_components} │ Nets: {n_nets} │ Directives: {n_directives}"
                )
                output_lines.append("─" * 50)
            else:
                output_lines.append(f"=== Circuit: {self.name} ({engine}) ===")
                output_lines.append(
                    f"Components: {n_components} | Nets: {n_nets} | Directives: {n_directives}"
                )
                output_lines.append("-" * 50)

        # Netlist with optional line numbers
        max_line_num = len(lines)
        num_width = len(str(max_line_num))

        for i, line in enumerate(lines, start=1):
            colored = colorize_line(line)
            if line_numbers:
                if syntax_highlight:
                    output_lines.append(f"{COMMENT}{i:>{num_width}} │{RESET} {colored}")
                else:
                    output_lines.append(f"{i:>{num_width}} | {colored}")
            else:
                output_lines.append(colored)

        return "\n".join(output_lines)

    def save_netlist(self, path: str | Path) -> Path:
        """Persist the netlist to ``path`` and return the resolved ``Path``."""

        p = Path(path)
        p.write_text(self.build_netlist(), encoding="utf-8")
        return p

    # ----------------------------------------------------------------------------------
    # Hash (deterministic) - part of M1 contract
    # ----------------------------------------------------------------------------------
    def hash(self, *, extra: dict[str, object] | None = None) -> str:  # pragma: no cover - thin
        """Return a deterministic short hash for this circuit.

        Wrapper around ``spicelab.core.types.circuit_hash`` so callers do not need
        to import the helper directly. ``extra`` can include engine/version/analysis
        args to bind caches firmly to execution context.
        """
        from .types import circuit_hash  # local import to avoid cycle during module init

        return circuit_hash(self, extra=extra)

    # ----------------------------------------------------------------------------------
    # Validation (M4 DX improvement)
    # ----------------------------------------------------------------------------------
    def validate(self, strict: bool = False) -> ValidationResult:
        """Validate circuit topology and component values.

        Performs checks:
        - Ground reference exists
        - No floating nodes (connected to only one component)
        - No unusual component values
        - No voltage source shorts (parallel voltage sources)

        Args:
            strict: If True, treat warnings as errors

        Returns:
            ValidationResult with errors and warnings

        Example:
            >>> result = circuit.validate()
            >>> if result.has_issues():
            ...     print(result)
            >>> if not result.is_valid:
            ...     raise ValueError("Circuit has errors")
        """
        from ..validators.circuit_validation import validate_circuit

        return validate_circuit(self, strict=strict)

    # ----------------------------------------------------------------------------------
    # Introspection helpers
    # ----------------------------------------------------------------------------------
    def _net_label(self, net: Net | None) -> str:
        if net is None:
            return "<unconnected>"

        # Use canonical net for merged nets
        canonical = self._get_canonical_net(net)

        if canonical is GND or getattr(canonical, "name", None) == "0":
            return "0"
        if getattr(canonical, "name", None):
            return str(canonical.name)
        node_id = self._net_ids.get(canonical)
        if node_id is None:
            self._assign_node_ids()
            node_id = self._net_ids.get(canonical)
        return f"N{node_id:03d}" if node_id is not None else "<unnamed>"

    def summary(self) -> str:
        """Return a human-readable summary of the circuit and connectivity."""

        self._assign_node_ids()

        lines: list[str] = []
        warnings: list[str] = []

        lines.append(f"Circuit: {self.name}")
        lines.append(f"Components ({len(self._components)}):")

        for comp in self._components:
            port_descriptions: list[str] = []
            for port in comp.ports:
                net = self._port_to_net.get(port)
                label = self._port_labels.get(port) or self._net_label(net)
                if label == "<unconnected>":
                    warnings.append(f"Port {comp.ref}.{port.name} is unconnected")
                port_descriptions.append(f"{port.name}->{label}")
            port_info = ", ".join(port_descriptions) if port_descriptions else "<no ports>"
            lines.append(f"  - {comp.ref} ({type(comp).__name__}): {port_info}")

        net_names = sorted({self._net_label(net) for net in self._port_to_net.values() if net})
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
        """Return a Graphviz DOT representation of the circuit."""

        self._assign_node_ids()

        lines: list[str] = ["graph circuit {", "  rankdir=LR;"]

        comp_ids: dict[Component, str] = {}
        for idx, comp in enumerate(self._components, start=1):
            comp_id = f"comp_{idx}"
            comp_ids[comp] = comp_id
            label = f"{comp.ref}\\n{type(comp).__name__}"
            lines.append(f'  "{comp_id}" [shape=box,label="{label}"];')

        net_ids: dict[Net | None, str] = {}
        net_counter = 1

        def _net_node(net: Net | None) -> str:
            nonlocal net_counter
            if net in net_ids:
                return net_ids[net]
            node_id = f"net_{net_counter}"
            net_counter += 1
            label = self._net_label(net)
            shape = "ellipse" if label != "<unconnected>" else "point"
            net_ids[net] = node_id
            lines.append(f'  "{node_id}" [shape={shape},label="{label}"];')
            return node_id

        for comp in self._components:
            comp_id = comp_ids[comp]
            for port in comp.ports:
                net = self._port_to_net.get(port)
                net_id = _net_node(net)
                lines.append(f'  "{comp_id}" -- "{net_id}" [label="{port.name}",fontsize=10];')

        lines.append("}")
        return "\n".join(lines)

    # ----------------------------------------------------------------------------------
    # Notebook-friendly helpers
    # ----------------------------------------------------------------------------------
    def connectivity_dataframe(self, *, sort: bool = True, include_type: bool = True):
        """Return a pandas DataFrame describing component/net connectivity.

        Columns: ``component``, ``type`` (optional), ``port`` and ``net``. The
        returned DataFrame is ideal for Jupyter notebooks where an interactive
        table is easier to scan than the plain text summary.
        """

        try:
            import pandas as pd
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("pandas is required for connectivity_dataframe()") from exc

        self._assign_node_ids()

        rows: list[dict[str, object]] = []
        for comp in self._components:
            for order, port in enumerate(comp.ports):
                net = self._port_to_net.get(port)
                label = self._port_labels.get(port) or self._net_label(net)
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
        """Return a fixed-width table describing component connections."""

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
    # Netlist import
    # ----------------------------------------------------------------------------------
    @classmethod
    def from_netlist(cls, path: str | Path) -> Circuit:
        """Load a circuit from a plain SPICE netlist file."""

        p = Path(path)
        text = p.read_text(encoding="utf-8")

        from spicelab.io.spice_parser import parse_lines_to_ast, preprocess_netlist

        lines = preprocess_netlist(text)
        ast = parse_lines_to_ast(lines)

        name = p.stem
        if lines and lines[0].lstrip().startswith("*"):
            maybe_title = lines[0].lstrip()[1:].strip()
            if maybe_title:
                name = maybe_title

        circ = cls(name=name)

        for node in ast:
            kind_obj = node.get("type")
            kind = str(kind_obj) if isinstance(kind_obj, str) else None
            raw_obj = node.get("raw")
            raw = str(raw_obj) if isinstance(raw_obj, str) else ""

            if kind == "subckt":
                header = raw.splitlines()[0]
                parts = header.split()
                if len(parts) >= 2:
                    circ._subckt_defs[parts[1]] = raw
                circ.add_directive(raw)
                continue

            if kind == "comment" or kind == "directive":
                circ.add_directive(raw)
                continue

            if kind != "component":
                continue

            tokens_obj = node.get("tokens")
            tokens: list[str] = [str(t) for t in tokens_obj] if isinstance(tokens_obj, list) else []
            if not tokens:
                continue

            card = tokens[0]
            letter = cast(str | None, node.get("letter"))
            ref = cast(str | None, node.get("ref"))

            try:
                comp = cls._component_from_tokens(letter, ref, tokens)
                if comp is None:
                    circ.add_directive(raw)
                    continue
                circ.add(comp)
                circ._connect_from_tokens(comp, tokens[1:])
                if letter == "X":
                    circ._subckt_instances.append(
                        {"inst": card, "subckt": tokens[-1], "tokens": tokens}
                    )
            except Exception as exc:  # pragma: no cover - defensive fallback
                log.warning("Failed to parse component '%s': %s", card, exc)
                circ.add_directive(raw)

        return circ

    # ----------------------------------------------------------------------------------
    # Helpers for from_netlist
    # ----------------------------------------------------------------------------------
    @staticmethod
    def _component_from_tokens(
        letter: str | None, ref: str | None, tokens: list[str]
    ) -> Component | None:
        if not letter or not ref:
            return None

        letter = letter.upper()
        value = " ".join(tokens[3:]) if len(tokens) > 3 else ""

        if letter == "R":
            return Resistor(ref=ref, value=value)
        if letter == "C":
            return Capacitor(ref=ref, value=value)
        if letter == "L":
            return Inductor(ref=ref, value=value)
        if letter == "V":
            return Vdc(ref=ref, value=value)
        if letter == "I":
            return Idc(ref=ref, value=value)
        if letter == "E":
            gain = " ".join(tokens[5:]) if len(tokens) > 5 else ""
            return VCVS(ref=ref, gain=gain)
        if letter == "G":
            gm = " ".join(tokens[5:]) if len(tokens) > 5 else ""
            return VCCS(ref=ref, gm=gm)
        if letter == "F":
            gain = " ".join(tokens[4:]) if len(tokens) > 4 else ""
            ctrl = tokens[3] if len(tokens) > 3 else ""
            return CCCS(ref=ref, ctrl_vsrc=ctrl, gain=gain)
        if letter == "H":
            r = " ".join(tokens[4:]) if len(tokens) > 4 else ""
            ctrl = tokens[3] if len(tokens) > 3 else ""
            return CCVS(ref=ref, ctrl_vsrc=ctrl, r=r)
        if letter == "D":
            model = tokens[3] if len(tokens) > 3 else ""
            return Diode(ref=ref, model=model)
        if letter == "S":
            model = " ".join(tokens[5:]) if len(tokens) > 5 else ""
            return VSwitch(ref=ref, model=model)
        if letter == "W":
            model = " ".join(tokens[4:]) if len(tokens) > 4 else ""
            ctrl = tokens[3] if len(tokens) > 3 else ""
            return ISwitch(ref=ref, ctrl_vsrc=ctrl, model=model)

        # Subcircuits and unsupported devices are preserved as directives
        return None

    def _connect_from_tokens(self, component: Component, node_tokens: list[str]) -> None:
        port_iter = iter(component.ports)
        for node_name in node_tokens:
            try:
                port = next(port_iter)
            except StopIteration:
                break
            net = self._get_or_create_net(node_name)
            self.connect(port, net)

    def _get_or_create_net(self, name: str) -> Net:
        if name == "0":
            return GND
        for net in self._port_to_net.values():
            if getattr(net, "name", None) == name:
                return net
        new = Net(name)
        return new
