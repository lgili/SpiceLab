# mypy: ignore-errors
"""Convert parsed ASC data to SpiceLab Circuit objects.

This module converts the result from the ASC parser into SpiceLab Circuit
objects with proper component instances and connections.

Features:
- Maps LTspice symbols to SpiceLab components
- Handles wiring using geometric analysis (wire intersections)
- Emits warnings for unsupported symbols/components
- Adds SPICE directives (parameters, analysis commands) to the circuit

Example
-------
>>> from spicelab.io import parse_asc_file
>>> from spicelab.io.asc_converter import asc_to_circuit
>>> result = parse_asc_file("circuit.asc")
>>> circuit, warnings = asc_to_circuit(result)
>>> if warnings:
...     for w in warnings:
...         print(f"WARNING: {w}")
>>> print(circuit.build_netlist())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..core.circuit import Circuit
from ..core.components import (
    CCCS,
    CCVS,
    VCCS,
    VCVS,
    Capacitor,
    Component,
    Diode,
    Idc,
    Inductor,
    OpAmpIdeal,
    Resistor,
    Vdc,
)
from ..core.net import GND, Net
from .asc_parser import AscParseResult, SymbolComponent, Wire

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conversion Result
# ---------------------------------------------------------------------------


@dataclass
class ConversionWarning:
    """Warning generated during ASC to Circuit conversion."""

    category: str
    message: str
    component_ref: str | None = None
    symbol: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        prefix = ""
        if self.component_ref:
            prefix = f"[{self.component_ref}] "
        elif self.symbol:
            prefix = f"[{self.symbol}] "
        return f"{self.category}: {prefix}{self.message}"


@dataclass
class ConversionResult:
    """Result of ASC to Circuit conversion."""

    circuit: Circuit
    warnings: list[ConversionWarning] = field(default_factory=list)
    converted_components: list[str] = field(default_factory=list)
    skipped_components: list[str] = field(default_factory=list)
    # Mapping from ASC component ref to SpiceLab component
    component_map: dict[str, Component] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """True if no errors occurred (warnings are OK)."""
        return len(self.skipped_components) == 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def summary(self) -> dict[str, Any]:
        return {
            "converted": len(self.converted_components),
            "skipped": len(self.skipped_components),
            "warnings": len(self.warnings),
            "success": self.success,
        }


# ---------------------------------------------------------------------------
# Symbol to Component Mapping
# ---------------------------------------------------------------------------


# Map of LTspice symbol names to factory functions
# Factory signature: (ref: str, asc_comp: SymbolComponent) -> Component | None
SYMBOL_MAP: dict[str, type | None] = {
    # Passive components
    "res": Resistor,
    "res2": Resistor,
    "cap": Capacitor,
    "cap2": Capacitor,
    "polcap": Capacitor,
    "ind": Inductor,
    "ind2": Inductor,
    # Voltage sources
    "voltage": Vdc,
    # Current sources
    "current": Idc,
    # Diodes
    "diode": Diode,
    "dio": Diode,
    "zener": Diode,
    "schottky": Diode,
    "led": Diode,
    # Controlled sources
    "e": VCVS,
    "e2": VCVS,
    "g": VCCS,
    "g2": VCCS,
    "f": CCCS,
    "h": CCVS,
    # Op-amps (various LTspice names)
    "opamp": OpAmpIdeal,
    "opamp2": OpAmpIdeal,
    "UniversalOpamp": OpAmpIdeal,
    "UniversalOpAmp": OpAmpIdeal,
    "UniversalOpAmp2": OpAmpIdeal,
}

# Symbols that we know about but cannot convert (for better warnings)
KNOWN_UNSUPPORTED: dict[str, str] = {
    # Transistors
    "npn": "BJT NPN transistor - not yet implemented",
    "pnp": "BJT PNP transistor - not yet implemented",
    "nmos": "NMOS transistor - not yet implemented",
    "pmos": "PMOS transistor - not yet implemented",
    "nmos4": "NMOS 4-terminal transistor - not yet implemented",
    "pmos4": "PMOS 4-terminal transistor - not yet implemented",
    "njf": "N-channel JFET - not yet implemented",
    "pjf": "P-channel JFET - not yet implemented",
    # Transformers
    "ltrans": "Transformer - not yet implemented",
    "transformer": "Transformer - not yet implemented",
    # Transmission lines
    "tline": "Transmission line - not yet implemented",
    "ltline": "Lossy transmission line - not yet implemented",
    # Digital
    "buf": "Digital buffer - not yet implemented",
    "inv": "Digital inverter - not yet implemented",
    "and": "Digital AND gate - not yet implemented",
    "or": "Digital OR gate - not yet implemented",
    "xor": "Digital XOR gate - not yet implemented",
    "dflop": "D flip-flop - not yet implemented",
    "srflop": "SR flip-flop - not yet implemented",
    # Misc
    "bv": "Behavioral voltage source - not yet implemented",
    "bi": "Behavioral current source - not yet implemented",
    "mesfet": "MESFET - not yet implemented",
}


def _normalize_symbol(symbol: str) -> str:
    """Normalize symbol name for lookup."""
    # Remove path prefixes like 'OpAmps\\' or 'Misc/'
    normalized = symbol.replace("\\", "/")
    basename = normalized.split("/")[-1]
    return basename.lower()


def _create_component(
    asc_comp: SymbolComponent,
    warnings: list[ConversionWarning],
) -> Component | None:
    """Create a SpiceLab component from an ASC symbol component."""
    symbol = asc_comp.symbol
    normalized = _normalize_symbol(symbol)
    ref = asc_comp.ref or "?"
    value = asc_comp.value

    # Try direct lookup first
    comp_class = SYMBOL_MAP.get(normalized)
    if comp_class is None:
        # Try with original basename
        basename = asc_comp.symbol_basename
        comp_class = SYMBOL_MAP.get(basename)

    if comp_class is None:
        # Try case-insensitive search
        for key, cls in SYMBOL_MAP.items():
            if key.lower() == normalized:
                comp_class = cls
                break

    if comp_class is None:
        # Check if it's a known unsupported symbol
        reason = KNOWN_UNSUPPORTED.get(normalized)
        if reason is None:
            # Check basename too
            reason = KNOWN_UNSUPPORTED.get(asc_comp.symbol_basename.lower())

        if reason:
            warnings.append(
                ConversionWarning(
                    category="UNSUPPORTED_SYMBOL",
                    message=reason,
                    component_ref=ref,
                    symbol=symbol,
                )
            )
        else:
            warnings.append(
                ConversionWarning(
                    category="UNKNOWN_SYMBOL",
                    message=f"Unknown symbol '{symbol}' - cannot convert",
                    component_ref=ref,
                    symbol=symbol,
                )
            )
        return None

    # Create the component based on its type
    try:
        if comp_class == Resistor:
            return Resistor(ref=ref, value=value or "1k")
        elif comp_class == Capacitor:
            return Capacitor(ref=ref, value=value or "1u")
        elif comp_class == Inductor:
            return Inductor(ref=ref, value=value or "1m")
        elif comp_class == Vdc:
            return Vdc(ref=ref, value=value or "0")
        elif comp_class == Idc:
            return Idc(ref=ref, value=value or "0")
        elif comp_class == Diode:
            model = asc_comp.model or value or "D"
            return Diode(ref=ref, model=model)
        elif comp_class == VCVS:
            return VCVS(ref=ref, gain=value or "1")
        elif comp_class == VCCS:
            return VCCS(ref=ref, gm=value or "1")
        elif comp_class == CCCS:
            ctrl = asc_comp.attributes.get("Ctrl", "V1")
            return CCCS(ref=ref, ctrl_vsrc=ctrl, gain=value or "1")
        elif comp_class == CCVS:
            ctrl = asc_comp.attributes.get("Ctrl", "V1")
            return CCVS(ref=ref, ctrl_vsrc=ctrl, r=value or "1")
        elif comp_class == OpAmpIdeal:
            gain = value or "1e6"
            return OpAmpIdeal(ref=ref, gain=gain)
        else:
            warnings.append(
                ConversionWarning(
                    category="CONVERSION_ERROR",
                    message=f"No factory for component class {comp_class.__name__}",
                    component_ref=ref,
                    symbol=symbol,
                )
            )
            return None
    except Exception as e:
        warnings.append(
            ConversionWarning(
                category="CONVERSION_ERROR",
                message=f"Failed to create component: {e}",
                component_ref=ref,
                symbol=symbol,
            )
        )
        return None


# ---------------------------------------------------------------------------
# Pin Position Calculation
# ---------------------------------------------------------------------------

# Standard pin offsets for different symbols (in LTspice grid units)
# These are relative to the symbol position, before rotation
# LTspice uses a 16-unit grid, typical component spacing is 80-112 units
PIN_OFFSETS: dict[str, list[tuple[int, int]]] = {
    # 2-terminal passives (vertical orientation R0)
    "res": [(0, 0), (0, 80)],
    "cap": [(0, 0), (0, 64)],
    "ind": [(0, 0), (0, 80)],
    "voltage": [(0, 0), (0, 112)],
    "current": [(0, 0), (0, 112)],
    "diode": [(0, 0), (0, 64)],
    "dio": [(0, 0), (0, 64)],
    # Op-amp (3 pins: inp, inn, out)
    # Standard LTspice UniversalOpAmp pinout
    "opamp": [(-32, -32), (-32, 32), (32, 0)],  # inp (+), inn (-), out
    "opamp2": [(-32, -32), (-32, 32), (32, 0)],
    "universalopamp": [(-32, -32), (-32, 32), (32, 0)],
    "universalopamp2": [(-32, -32), (-32, 32), (32, 0)],
    # 4-terminal controlled sources
    "e": [(0, 0), (0, 80), (16, 16), (16, 64)],  # p, n, cp, cn
    "g": [(0, 0), (0, 80), (16, 16), (16, 64)],
    "f": [(0, 0), (0, 80)],  # 2 terminal + ctrl ref
    "h": [(0, 0), (0, 80)],
}

# Default 2-terminal pin offsets
DEFAULT_2PIN = [(0, 0), (0, 80)]


def _rotate_point(x: int, y: int, rotation: str) -> tuple[int, int]:
    """Apply rotation to a point relative to origin."""
    if rotation == "R0":
        return (x, y)
    elif rotation == "R90":
        return (-y, x)
    elif rotation == "R180":
        return (-x, -y)
    elif rotation == "R270":
        return (y, -x)
    elif rotation == "M0":  # Mirror horizontal
        return (-x, y)
    elif rotation == "M90":
        return (-y, -x)
    elif rotation == "M180":
        return (x, -y)
    elif rotation == "M270":
        return (y, x)
    else:
        return (x, y)


def _get_pin_positions(asc_comp: SymbolComponent) -> list[tuple[int, int]]:
    """Get absolute pin positions for a component."""
    symbol = _normalize_symbol(asc_comp.symbol)
    offsets = PIN_OFFSETS.get(symbol, DEFAULT_2PIN)

    # Also try basename
    if symbol not in PIN_OFFSETS:
        basename = asc_comp.symbol_basename.lower()
        offsets = PIN_OFFSETS.get(basename, DEFAULT_2PIN)

    positions = []
    for ox, oy in offsets:
        rx, ry = _rotate_point(ox, oy, asc_comp.rotation)
        positions.append((asc_comp.x + rx, asc_comp.y + ry))

    return positions


# ---------------------------------------------------------------------------
# Wire Network Analysis
# ---------------------------------------------------------------------------


class _UnionFind:
    """Simple union-find for net connectivity."""

    def __init__(self) -> None:
        self._parent: dict[tuple[int, int], tuple[int, int]] = {}
        self._labels: dict[tuple[int, int], str] = {}

    def add(self, point: tuple[int, int], label: str | None = None) -> None:
        if point not in self._parent:
            self._parent[point] = point
        if label and point not in self._labels:
            self._labels[point] = label

    def find(self, point: tuple[int, int]) -> tuple[int, int]:
        if point not in self._parent:
            self._parent[point] = point
            return point
        if self._parent[point] != point:
            self._parent[point] = self.find(self._parent[point])
        return self._parent[point]

    def union(self, a: tuple[int, int], b: tuple[int, int]) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Prefer labeled root
        if ra in self._labels:
            self._parent[rb] = ra
        else:
            self._parent[ra] = rb
            if rb in self._labels and ra not in self._labels:
                pass  # rb stays root with label
            elif ra in self._labels:
                self._labels[rb] = self._labels[ra]

    def get_label(self, point: tuple[int, int]) -> str | None:
        root = self.find(point)
        return self._labels.get(root)


def _points_on_wire(wire: Wire) -> list[tuple[int, int]]:
    """Get all grid points on a wire (assumes axis-aligned)."""
    points = []
    x1, y1, x2, y2 = wire.x1, wire.y1, wire.x2, wire.y2

    if x1 == x2:  # Vertical
        y_min, y_max = min(y1, y2), max(y1, y2)
        for y in range(y_min, y_max + 1, 16):  # LTspice grid is 16
            points.append((x1, y))
        # Ensure endpoints are included
        if (x1, y1) not in points:
            points.append((x1, y1))
        if (x2, y2) not in points:
            points.append((x2, y2))
    elif y1 == y2:  # Horizontal
        x_min, x_max = min(x1, x2), max(x1, x2)
        for x in range(x_min, x_max + 1, 16):
            points.append((x, y1))
        if (x1, y1) not in points:
            points.append((x1, y1))
        if (x2, y2) not in points:
            points.append((x2, y2))
    else:
        # Diagonal - just endpoints
        points = [(x1, y1), (x2, y2)]

    return points


def _build_net_map(
    asc_result: AscParseResult,
    component_pins: dict[str, list[tuple[int, int]]],
) -> tuple[_UnionFind, dict[tuple[int, int], str]]:
    """Build union-find structure for net connectivity."""
    uf = _UnionFind()

    # Add all wire points
    for wire in asc_result.wires:
        p1 = (wire.x1, wire.y1)
        p2 = (wire.x2, wire.y2)
        uf.add(p1)
        uf.add(p2)
        uf.union(p1, p2)

        # Also union all intermediate points on axis-aligned wires
        points = _points_on_wire(wire)
        for p in points:
            uf.add(p)
            uf.union(p1, p)

    # Add flags (labels)
    flag_positions: dict[tuple[int, int], str] = {}
    for flag in asc_result.flags:
        pos = (flag.x, flag.y)
        uf.add(pos, label=flag.name)
        flag_positions[pos] = flag.name

    # Add component pins
    for _ref, pins in component_pins.items():
        for pin in pins:
            uf.add(pin)

    return uf, flag_positions


def _find_net_for_pin(
    pin: tuple[int, int],
    uf: _UnionFind,
    nets: dict[tuple[int, int], Net],
    net_counter: list[int],
) -> Net:
    """Find or create the net for a pin position."""
    root = uf.find(pin)

    if root in nets:
        return nets[root]

    # Check for label
    label = uf.get_label(pin)
    if label:
        if label in ("0", "GND", "gnd"):
            nets[root] = GND
            return GND
        net = Net(label)
        nets[root] = net
        return net

    # Create auto-named net
    net_counter[0] += 1
    net = Net(f"N{net_counter[0]:03d}")
    nets[root] = net
    return net


# ---------------------------------------------------------------------------
# Main Conversion Function
# ---------------------------------------------------------------------------


def asc_to_circuit(
    asc_result: AscParseResult,
    circuit_name: str | None = None,
) -> ConversionResult:
    """Convert parsed ASC data to a SpiceLab Circuit.

    Parameters
    ----------
    asc_result : AscParseResult
        Parsed ASC file data
    circuit_name : str, optional
        Name for the circuit. If not provided, uses the filename or "imported"

    Returns
    -------
    ConversionResult
        Contains the circuit, warnings, and conversion statistics

    Example
    -------
    >>> result = parse_asc_file("circuit.asc")
    >>> conv = asc_to_circuit(result)
    >>> if conv.warnings:
    ...     for w in conv.warnings:
    ...         print(f"WARNING: {w}")
    >>> print(conv.circuit.build_netlist())
    """
    warnings: list[ConversionWarning] = []
    converted: list[str] = []
    skipped: list[str] = []
    component_map: dict[str, Component] = {}

    # Determine circuit name
    if circuit_name is None:
        if asc_result.file_path:
            from pathlib import Path

            circuit_name = Path(asc_result.file_path).stem
        else:
            circuit_name = "imported"

    circuit = Circuit(circuit_name)

    # First pass: create components and get pin positions
    component_pins: dict[str, list[tuple[int, int]]] = {}

    for asc_comp in asc_result.components:
        ref = asc_comp.ref or f"X{len(component_map)}"

        # Create the SpiceLab component
        comp = _create_component(asc_comp, warnings)

        if comp is None:
            skipped.append(ref)
            continue

        # Get pin positions
        pins = _get_pin_positions(asc_comp)
        component_pins[ref] = pins

        # Add to circuit
        circuit.add(comp)
        component_map[ref] = comp
        converted.append(ref)

    # Build net connectivity map
    uf, flag_positions = _build_net_map(asc_result, component_pins)

    # Second pass: connect components to nets
    nets: dict[tuple[int, int], Net] = {}
    net_counter = [0]

    for ref, comp in component_map.items():
        pins = component_pins.get(ref, [])

        if len(pins) != len(comp.ports):
            warnings.append(
                ConversionWarning(
                    category="PIN_MISMATCH",
                    message=(
                        f"Pin count mismatch: symbol has {len(pins)} pins, "
                        f"component has {len(comp.ports)} ports"
                    ),
                    component_ref=ref,
                )
            )
            # Try to connect what we can
            pins = pins[: len(comp.ports)]

        for port, pin in zip(comp.ports, pins, strict=False):
            net = _find_net_for_pin(pin, uf, nets, net_counter)
            circuit.connect(port, net)

    # Add SPICE directives from the ASC file
    for name, param in asc_result.parameters.items():
        circuit.add_directive(f".param {name}={param.value}")

    for cmd in asc_result.analysis_commands:
        circuit.add_directive(f".{cmd.analysis_type} {cmd.parameters}")

    # Add measurements as comments (they need to be handled separately in simulation)
    if asc_result.measurements:
        circuit.add_directive("* Measurements from ASC file:")
        for meas in asc_result.measurements:
            circuit.add_directive(f".meas {meas.name} {meas.measurement_type} {meas.expression}")

    return ConversionResult(
        circuit=circuit,
        warnings=warnings,
        converted_components=converted,
        skipped_components=skipped,
        component_map=component_map,
    )


def load_circuit_from_asc(path: str) -> ConversionResult:
    """Load and convert an ASC file to a SpiceLab Circuit.

    This is a convenience function that combines parsing and conversion.

    Parameters
    ----------
    path : str
        Path to the .asc file

    Returns
    -------
    ConversionResult
        Contains the circuit, warnings, and conversion statistics

    Example
    -------
    >>> result = load_circuit_from_asc("circuit.asc")
    >>> if result.has_warnings:
    ...     for w in result.warnings:
    ...         print(f"WARNING: {w}")
    >>> circuit = result.circuit
    """
    from .asc_parser import parse_asc_file

    asc_result = parse_asc_file(path)
    return asc_to_circuit(asc_result)


def print_conversion_result(result: ConversionResult) -> None:
    """Print a summary of the conversion result."""
    print("ASC to Circuit Conversion")
    print("=" * 50)
    print(f"Circuit name: {result.circuit.name}")
    print(f"Converted: {len(result.converted_components)} components")
    print(f"Skipped: {len(result.skipped_components)} components")
    print(f"Warnings: {len(result.warnings)}")
    print()

    if result.converted_components:
        print("Converted components:")
        for ref in result.converted_components:
            comp = result.component_map.get(ref)
            if comp:
                print(f"  {ref}: {type(comp).__name__} = {comp.value}")
        print()

    if result.skipped_components:
        print("Skipped components:")
        for ref in result.skipped_components:
            print(f"  {ref}")
        print()

    if result.warnings:
        print("Warnings:")
        for w in result.warnings:
            print(f"  {w}")
        print()


__all__ = [
    "ConversionWarning",
    "ConversionResult",
    "asc_to_circuit",
    "load_circuit_from_asc",
    "print_conversion_result",
    "SYMBOL_MAP",
    "KNOWN_UNSUPPORTED",
]
