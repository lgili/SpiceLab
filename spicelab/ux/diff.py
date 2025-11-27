"""Circuit diff tool for visualizing changes.

Provides comparison between circuit states or different circuits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.circuit import Circuit
    from ..core.components import Component


class ChangeType(Enum):
    """Type of change between circuits."""

    ADDED = auto()
    REMOVED = auto()
    MODIFIED = auto()
    UNCHANGED = auto()


@dataclass
class DiffChange:
    """A single change between two circuits.

    Attributes:
        type: Type of change
        category: Category of change (component, connection, directive)
        old: Previous value (None for added)
        new: New value (None for removed)
        details: Additional details about the change
    """

    type: ChangeType
    category: str
    old: Any = None
    new: Any = None
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        symbols = {
            ChangeType.ADDED: "+",
            ChangeType.REMOVED: "-",
            ChangeType.MODIFIED: "~",
            ChangeType.UNCHANGED: " ",
        }
        symbol = symbols.get(self.type, "?")

        if self.type == ChangeType.ADDED:
            return f"{symbol} [{self.category}] Added: {self.new}"
        elif self.type == ChangeType.REMOVED:
            return f"{symbol} [{self.category}] Removed: {self.old}"
        elif self.type == ChangeType.MODIFIED:
            return f"{symbol} [{self.category}] Modified: {self.old} -> {self.new}"
        else:
            return f"{symbol} [{self.category}] {self.old}"


@dataclass
class CircuitDiff:
    """Result of comparing two circuits.

    Attributes:
        circuit_a: Name of first circuit
        circuit_b: Name of second circuit
        changes: List of changes
    """

    circuit_a: str
    circuit_b: str
    changes: list[DiffChange] = field(default_factory=list)

    @property
    def added(self) -> list[DiffChange]:
        """Get all added items."""
        return [c for c in self.changes if c.type == ChangeType.ADDED]

    @property
    def removed(self) -> list[DiffChange]:
        """Get all removed items."""
        return [c for c in self.changes if c.type == ChangeType.REMOVED]

    @property
    def modified(self) -> list[DiffChange]:
        """Get all modified items."""
        return [c for c in self.changes if c.type == ChangeType.MODIFIED]

    @property
    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return any(c.type != ChangeType.UNCHANGED for c in self.changes)

    def summary(self) -> str:
        """Get a summary of changes.

        Returns:
            Summary string
        """
        added = len(self.added)
        removed = len(self.removed)
        modified = len(self.modified)

        parts = []
        if added:
            parts.append(f"+{added} added")
        if removed:
            parts.append(f"-{removed} removed")
        if modified:
            parts.append(f"~{modified} modified")

        if not parts:
            return "No changes"
        return ", ".join(parts)

    def __str__(self) -> str:
        lines = [
            f"Circuit Diff: {self.circuit_a} -> {self.circuit_b}",
            "=" * 60,
            f"Summary: {self.summary()}",
            "-" * 60,
        ]

        # Group by category
        by_category: dict[str, list[DiffChange]] = {}
        for change in self.changes:
            if change.type != ChangeType.UNCHANGED:
                by_category.setdefault(change.category, []).append(change)

        for category, changes in sorted(by_category.items()):
            lines.append(f"\n{category.upper()}:")
            for change in changes:
                lines.append(f"  {change}")

        if not by_category:
            lines.append("\n(no changes)")

        return "\n".join(lines)

    def to_unified_diff(self) -> str:
        """Generate unified diff format output.

        Returns:
            Unified diff string
        """
        lines = [
            f"--- {self.circuit_a}",
            f"+++ {self.circuit_b}",
        ]

        for change in self.changes:
            if change.type == ChangeType.REMOVED:
                lines.append(f"- {change.old}")
            elif change.type == ChangeType.ADDED:
                lines.append(f"+ {change.new}")
            elif change.type == ChangeType.MODIFIED:
                lines.append(f"- {change.old}")
                lines.append(f"+ {change.new}")

        return "\n".join(lines)

    def to_html(self) -> str:
        """Generate HTML diff visualization.

        Returns:
            HTML string
        """
        css = """
        <style>
            .diff { font-family: monospace; }
            .diff-header { background: #f0f0f0; padding: 10px; }
            .diff-added { background: #e6ffec; color: #22863a; }
            .diff-removed { background: #ffebe9; color: #cb2431; }
            .diff-modified { background: #fff5b1; color: #735c0f; }
            .diff-category { font-weight: bold; margin-top: 10px; }
        </style>
        """

        html = [
            css,
            '<div class="diff">',
            '<div class="diff-header">',
            f"<strong>{self.circuit_a}</strong> → <strong>{self.circuit_b}</strong>",
            f"<br>Summary: {self.summary()}",
            "</div>",
        ]

        by_category: dict[str, list[DiffChange]] = {}
        for change in self.changes:
            if change.type != ChangeType.UNCHANGED:
                by_category.setdefault(change.category, []).append(change)

        for category, changes in sorted(by_category.items()):
            html.append(f'<div class="diff-category">{category}</div>')
            for change in changes:
                css_class = {
                    ChangeType.ADDED: "diff-added",
                    ChangeType.REMOVED: "diff-removed",
                    ChangeType.MODIFIED: "diff-modified",
                }.get(change.type, "")

                html.append(f'<div class="{css_class}">{change}</div>')

        html.append("</div>")
        return "\n".join(html)


def diff_circuits(
    circuit_a: Circuit,
    circuit_b: Circuit,
    *,
    ignore_names: bool = False,
    ignore_directives: bool = False,
) -> CircuitDiff:
    """Compare two circuits and return their differences.

    Args:
        circuit_a: First circuit (base)
        circuit_b: Second circuit (target)
        ignore_names: Ignore circuit name differences
        ignore_directives: Ignore directive differences

    Returns:
        CircuitDiff with all changes

    Example:
        >>> from spicelab.ux import diff_circuits
        >>> from spicelab.core.circuit import Circuit
        >>> from spicelab.core.components import Resistor
        >>>
        >>> c1 = Circuit("v1")
        >>> c1.add(Resistor(ref="1", resistance=1000))
        >>>
        >>> c2 = Circuit("v2")
        >>> c2.add(Resistor(ref="1", resistance=2000))
        >>> c2.add(Resistor(ref="2", resistance=3000))
        >>>
        >>> diff = diff_circuits(c1, c2)
        >>> print(diff)
        Circuit Diff: v1 -> v2
        ============================================================
        Summary: +1 added, ~1 modified
        ------------------------------------------------------------

        COMPONENT:
          ~ [component] Modified: R1(1000) -> R1(2000)
          + [component] Added: R2(3000)
    """
    changes: list[DiffChange] = []

    # Compare names
    if not ignore_names and circuit_a.name != circuit_b.name:
        changes.append(
            DiffChange(
                type=ChangeType.MODIFIED,
                category="name",
                old=circuit_a.name,
                new=circuit_b.name,
            )
        )

    # Compare components
    components_a = {c.ref: c for c in circuit_a._components}
    components_b = {c.ref: c for c in circuit_b._components}

    all_refs = set(components_a.keys()) | set(components_b.keys())

    for ref in sorted(all_refs):
        comp_a = components_a.get(ref)
        comp_b = components_b.get(ref)

        if comp_a is None and comp_b is not None:
            changes.append(
                DiffChange(
                    type=ChangeType.ADDED,
                    category="component",
                    new=_describe_component(comp_b),
                    details={"ref": ref, "component": comp_b},
                )
            )
        elif comp_a is not None and comp_b is None:
            changes.append(
                DiffChange(
                    type=ChangeType.REMOVED,
                    category="component",
                    old=_describe_component(comp_a),
                    details={"ref": ref, "component": comp_a},
                )
            )
        elif comp_a is not None and comp_b is not None:
            comp_changes = _compare_components(comp_a, comp_b)
            if comp_changes:
                changes.append(
                    DiffChange(
                        type=ChangeType.MODIFIED,
                        category="component",
                        old=_describe_component(comp_a),
                        new=_describe_component(comp_b),
                        details={"ref": ref, "changes": comp_changes},
                    )
                )

    # Compare connections
    conn_a = _get_connections(circuit_a)
    conn_b = _get_connections(circuit_b)

    all_ports = set(conn_a.keys()) | set(conn_b.keys())

    for port_key in sorted(all_ports):
        net_a = conn_a.get(port_key)
        net_b = conn_b.get(port_key)

        if net_a is None and net_b is not None:
            changes.append(
                DiffChange(
                    type=ChangeType.ADDED,
                    category="connection",
                    new=f"{port_key} -> {net_b}",
                )
            )
        elif net_a is not None and net_b is None:
            changes.append(
                DiffChange(
                    type=ChangeType.REMOVED,
                    category="connection",
                    old=f"{port_key} -> {net_a}",
                )
            )
        elif net_a != net_b:
            changes.append(
                DiffChange(
                    type=ChangeType.MODIFIED,
                    category="connection",
                    old=f"{port_key} -> {net_a}",
                    new=f"{port_key} -> {net_b}",
                )
            )

    # Compare directives
    if not ignore_directives:
        dir_a = set(circuit_a._directives)
        dir_b = set(circuit_b._directives)

        for d in sorted(dir_a - dir_b):
            changes.append(
                DiffChange(
                    type=ChangeType.REMOVED,
                    category="directive",
                    old=d,
                )
            )

        for d in sorted(dir_b - dir_a):
            changes.append(
                DiffChange(
                    type=ChangeType.ADDED,
                    category="directive",
                    new=d,
                )
            )

    return CircuitDiff(
        circuit_a=circuit_a.name,
        circuit_b=circuit_b.name,
        changes=changes,
    )


def _describe_component(comp: Component) -> str:
    """Create a short description of a component."""
    attrs = []

    # Get key attributes
    for attr in ["resistance", "capacitance", "inductance", "value", "gain", "gm"]:
        if hasattr(comp, attr):
            val = getattr(comp, attr)
            if val is not None:
                attrs.append(str(val))
                break

    if attrs:
        return f"{comp.ref}({', '.join(attrs)})"
    return f"{comp.ref}"


def _compare_components(comp_a: Component, comp_b: Component) -> dict[str, tuple[Any, Any]]:
    """Compare two components and return differences."""
    changes: dict[str, tuple[Any, Any]] = {}

    # Compare type
    if type(comp_a).__name__ != type(comp_b).__name__:
        changes["type"] = (type(comp_a).__name__, type(comp_b).__name__)

    # Compare common attributes
    attrs_a = _get_component_attrs(comp_a)
    attrs_b = _get_component_attrs(comp_b)

    all_attrs = set(attrs_a.keys()) | set(attrs_b.keys())
    for attr in all_attrs:
        val_a = attrs_a.get(attr)
        val_b = attrs_b.get(attr)
        if val_a != val_b:
            changes[attr] = (val_a, val_b)

    return changes


def _get_component_attrs(comp: Component) -> dict[str, Any]:
    """Get component attributes as a dictionary."""
    attrs = {}
    for key in dir(comp):
        if key.startswith("_") or key in ("ports", "ref"):
            continue
        try:
            value = getattr(comp, key)
            if not callable(value):
                attrs[key] = value
        except Exception:
            pass
    return attrs


def _get_connections(circuit: Circuit) -> dict[str, str | None]:
    """Get circuit connections as a dictionary."""
    connections: dict[str, str | None] = {}
    for port, net in circuit._port_to_net.items():
        key = f"{port.owner.ref}.{port.name}"
        net_name = getattr(net, "name", None)
        if net_name == "0" or net_name is None:
            connections[key] = "GND"
        else:
            connections[key] = net_name
    return connections


__all__ = [
    "ChangeType",
    "DiffChange",
    "CircuitDiff",
    "diff_circuits",
]
