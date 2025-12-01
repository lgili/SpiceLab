"""Circuit validation to detect common issues before simulation.

Checks for:
- Floating nodes (connected to only one component)
- Missing ground reference
- Short circuits (voltage sources in parallel)
- Unusual component values

This module provides pre-simulation validation to catch common mistakes
that would cause simulation failures or incorrect results.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.circuit import Circuit

__all__ = ["validate_circuit", "ValidationWarning", "ValidationResult"]


@dataclass
class ValidationWarning:
    """Warning about potential circuit issue."""

    severity: str  # "error", "warning", "info"
    message: str
    component_ref: str | None = None
    suggestion: str | None = None


@dataclass
class ValidationResult:
    """Result of circuit validation."""

    is_valid: bool
    errors: list[ValidationWarning]
    warnings: list[ValidationWarning]

    def has_issues(self) -> bool:
        """Check if there are any errors or warnings."""
        return len(self.errors) > 0 or len(self.warnings) > 0

    def __str__(self) -> str:
        """Human-readable validation report."""
        if not self.has_issues():
            return "✓ Circuit validation passed"

        lines = []
        if self.errors:
            lines.append("ERRORS:")
            for err in self.errors:
                comp = f" [{err.component_ref}]" if err.component_ref else ""
                lines.append(f"  ✗{comp} {err.message}")
                if err.suggestion:
                    lines.append(f"    → {err.suggestion}")

        if self.warnings:
            lines.append("WARNINGS:")
            for warn in self.warnings:
                comp = f" [{warn.component_ref}]" if warn.component_ref else ""
                lines.append(f"  ⚠{comp} {warn.message}")
                if warn.suggestion:
                    lines.append(f"    → {warn.suggestion}")

        return "\n".join(lines)


def validate_circuit(circuit: Circuit, strict: bool = False) -> ValidationResult:
    """Validate circuit topology and component values.

    Performs checks:
    - Ground reference exists
    - No floating nodes (connected to only one component)
    - No unusual component values
    - No voltage source shorts (parallel voltage sources)
    - No current source open circuits (series current sources)

    Args:
        circuit: Circuit to validate
        strict: If True, treat warnings as errors

    Returns:
        ValidationResult with errors and warnings

    Example:
        >>> result = validate_circuit(my_circuit)
        >>> if result.has_issues():
        ...     print(result)
        >>> if not result.is_valid:
        ...     raise ValueError("Circuit has errors")
    """
    errors: list[ValidationWarning] = []
    warnings: list[ValidationWarning] = []

    # Check 1: Ground reference exists
    has_ground = _check_ground_reference(circuit)
    if not has_ground:
        errors.append(
            ValidationWarning(
                severity="error",
                message="No ground (node 0) reference found",
                suggestion="Connect at least one component to GND for DC operating point",
            )
        )

    # Check 2: Floating nodes (only connected to one component port)
    floating_warnings = _check_floating_nodes(circuit)
    errors.extend(floating_warnings)

    # Check 3: Voltage source loops (parallel voltage sources)
    vsource_warnings = _check_voltage_source_loops(circuit)
    errors.extend(vsource_warnings)

    # Check 4: Current source loops (series current sources)
    isource_warnings = _check_current_source_loops(circuit)
    errors.extend(isource_warnings)

    # Check 5: Unusual component values
    value_warnings = _check_component_values(circuit)
    warnings.extend(value_warnings)

    # If strict mode, convert warnings to errors
    if strict:
        for warn in warnings:
            warn.severity = "error"
        errors.extend(warnings)
        warnings = []

    is_valid = len(errors) == 0

    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


def _check_ground_reference(circuit: Circuit) -> bool:
    """Check if circuit has ground reference (node 0 or GND).

    A circuit needs at least one connection to ground for SPICE to
    establish a reference voltage (node 0).
    """
    from ..core.net import GND

    # Check if any port is connected to GND
    for net in circuit._port_to_net.values():
        # Get canonical net (handles Union-Find merging)
        canonical = circuit._get_canonical_net(net)
        if canonical is GND:
            return True
        # Also check if net has name "0" or "GND"
        net_name = getattr(canonical, "name", None)
        if net_name in ("0", "GND", "gnd"):
            return True

    return False


def _check_floating_nodes(circuit: Circuit) -> list[ValidationWarning]:
    """Check for floating nodes (nets connected to only one component port).

    A floating node has no defined voltage because it's only connected to
    one component terminal. This will cause simulation failures.

    Returns:
        List of ValidationWarning for each floating node found
    """
    from ..core.net import GND

    warnings: list[ValidationWarning] = []

    # Build map of canonical net -> list of (component, port)
    net_connections: dict[Any, list[tuple[str, str]]] = defaultdict(list)

    for comp in circuit._components:
        for port in comp.ports:
            net = circuit._port_to_net.get(port)
            if net is not None:
                canonical = circuit._get_canonical_net(net)
                net_connections[canonical].append((comp.ref, port.name))

    # Check each net for floating (single connection)
    for net, connections in net_connections.items():
        # Skip ground - it's always valid
        if net is GND:
            continue
        net_name = getattr(net, "name", None)
        if net_name in ("0", "GND", "gnd"):
            continue

        # A net with only one connection is floating
        if len(connections) == 1:
            comp_ref, port_name = connections[0]
            net_label = net_name if net_name else f"unnamed net at {comp_ref}.{port_name}"

            warnings.append(
                ValidationWarning(
                    severity="error",
                    message=(
                        f"Floating node: {net_label} " f"(only connected to {comp_ref}.{port_name})"
                    ),
                    component_ref=comp_ref,
                    suggestion="Connect this node to another component or to ground",
                )
            )

    return warnings


def _check_voltage_source_loops(circuit: Circuit) -> list[ValidationWarning]:
    """Check for voltage source loops (parallel voltage sources).

    Two voltage sources connected in parallel create a conflict that
    SPICE cannot resolve, causing simulation failure.

    Returns:
        List of ValidationWarning for each voltage source loop found
    """

    warnings: list[ValidationWarning] = []

    # Find all voltage sources
    voltage_sources: list[tuple[str, Any, Any]] = []  # (ref, net+, net-)

    for comp in circuit._components:
        comp_type = type(comp).__name__
        # Voltage sources: Vdc, Vac, Vpulse, Vsin, etc.
        if comp_type.startswith("V") or comp_type in (
            "Vdc",
            "Vac",
            "Vpulse",
            "Vsin",
            "VsinT",
            "VpwlT",
        ):
            if len(comp.ports) >= 2:
                net_p = circuit._port_to_net.get(comp.ports[0])
                net_n = circuit._port_to_net.get(comp.ports[1])
                if net_p is not None and net_n is not None:
                    canon_p = circuit._get_canonical_net(net_p)
                    canon_n = circuit._get_canonical_net(net_n)
                    voltage_sources.append((comp.ref, canon_p, canon_n))

    # Check for parallel voltage sources (same nodes)
    seen_pairs: dict[tuple[Any, Any], str] = {}
    for ref, net_p, net_n in voltage_sources:
        # Normalize pair (smaller id first)
        pair = (net_p, net_n) if id(net_p) < id(net_n) else (net_n, net_p)

        if pair in seen_pairs:
            other_ref = seen_pairs[pair]
            warnings.append(
                ValidationWarning(
                    severity="error",
                    message=f"Voltage source loop: {ref} and {other_ref} are in parallel",
                    component_ref=ref,
                    suggestion="Remove one voltage source or add a series resistance",
                )
            )
        else:
            seen_pairs[pair] = ref

    return warnings


def _check_current_source_loops(circuit: Circuit) -> list[ValidationWarning]:
    """Check for current source loops (series current sources at a node).

    Two current sources connected in series at a node where only they
    connect creates a conflict - the node has no path for current to flow
    except through those sources, violating KCL if currents differ.

    More specifically, if a node has exactly 2 connections and both are
    current source terminals, it's likely a series connection that will fail.

    Returns:
        List of ValidationWarning for each problematic configuration found
    """
    from ..core.net import GND

    warnings: list[ValidationWarning] = []

    # Build map of canonical net -> list of (component, port, is_current_source)
    net_connections: dict[Any, list[tuple[str, str, bool]]] = defaultdict(list)

    current_source_types = (
        "Idc",
        "Iac",
        "Ipulse",
        "Isin",
        "IsinT",
        "Ipwl",
        "IpwlT",
        "CCCS",  # Current-controlled current source
        "BCurrent",  # Behavioral current source
    )

    for comp in circuit._components:
        comp_type = type(comp).__name__
        is_isrc = comp_type in current_source_types or comp_type.startswith("I")

        for port in comp.ports:
            net = circuit._port_to_net.get(port)
            if net is not None:
                canonical = circuit._get_canonical_net(net)
                net_connections[canonical].append((comp.ref, port.name, is_isrc))

    # Check for nodes with only current source connections
    for net, connections in net_connections.items():
        # Skip ground
        if net is GND:
            continue
        net_name = getattr(net, "name", None)
        if net_name in ("0", "GND", "gnd"):
            continue

        # Check if all connections at this node are current sources
        isrc_connections = [c for c in connections if c[2]]
        non_isrc_connections = [c for c in connections if not c[2]]

        # If a node has 2+ current source connections and NO other connections,
        # it's a series current source situation (problematic)
        if len(isrc_connections) >= 2 and len(non_isrc_connections) == 0:
            refs = [c[0] for c in isrc_connections]
            net_label = net_name if net_name else "internal node"

            warnings.append(
                ValidationWarning(
                    severity="error",
                    message=(
                        f"Current source loop at {net_label}: "
                        f"{', '.join(refs)} are in series with no load path"
                    ),
                    component_ref=refs[0],
                    suggestion=(
                        "Add a parallel resistance or other load at this node, "
                        "or remove one current source"
                    ),
                )
            )

    return warnings


def _check_component_values(circuit: Circuit) -> list[ValidationWarning]:
    """Check for unusual component values that might be typos."""
    warnings: list[ValidationWarning] = []

    # Check resistor values
    for comp in circuit._components:
        comp_type = type(comp).__name__

        if comp_type == "Resistor" and hasattr(comp, "resistance"):
            R = getattr(comp, "resistance")
            if isinstance(R, int | float):
                if R < 0.001:  # < 1mΩ
                    warnings.append(
                        ValidationWarning(
                            severity="warning",
                            message=f"Unusually small resistance: {R}Ω",
                            component_ref=comp.ref,
                            suggestion="Typical range: 1Ω - 1GΩ. Check for unit error?",
                        )
                    )
                elif R > 1e9:  # > 1GΩ
                    warnings.append(
                        ValidationWarning(
                            severity="warning",
                            message=f"Unusually large resistance: {R}Ω",
                            component_ref=comp.ref,
                            suggestion="Consider using open circuit or higher impedance model",
                        )
                    )

        elif comp_type == "Capacitor" and hasattr(comp, "capacitance"):
            C = getattr(comp, "capacitance")
            if isinstance(C, int | float):
                if C < 1e-15:  # < 1fF
                    warnings.append(
                        ValidationWarning(
                            severity="warning",
                            message=f"Unusually small capacitance: {C}F",
                            component_ref=comp.ref,
                            suggestion="Typical range: 1pF - 1F. Check for unit error?",
                        )
                    )
                elif C > 1:  # > 1F
                    warnings.append(
                        ValidationWarning(
                            severity="warning",
                            message=f"Unusually large capacitance: {C}F",
                            component_ref=comp.ref,
                            suggestion="Supercapacitor? Typical range: 1pF - 1F",
                        )
                    )

    return warnings
