# mypy: ignore-errors
"""Convert parsed ASC data to SpiceLab Circuit objects.

This module converts the result from the ASC parser into SpiceLab Circuit
objects with proper component instances and connections.

Features:
- Maps LTspice symbols to SpiceLab components
- Handles wiring using geometric analysis (wire intersections)
- Emits warnings for unsupported symbols/components
- Adds SPICE directives (parameters, analysis commands) to the circuit
- Parses analysis commands (.tran, .ac, .dc) into AnalysisSpec objects
- Provides run_asc_simulation() for easy simulation with LTspice (default)

Example
-------
>>> from spicelab.io import load_circuit_from_asc, run_asc_simulation
>>> result = load_circuit_from_asc("circuit.asc")
>>> sim_result = run_asc_simulation("circuit.asc")
>>> print(sim_result.dataset())
"""

from __future__ import annotations

import logging
import re
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
# Offsets verified against actual LTspice ASC files (PT1000_circuit_1.asc)
PIN_OFFSETS: dict[str, list[tuple[int, int]]] = {
    # 2-terminal passives (vertical orientation R0)
    # Pin 0 = top terminal, Pin 1 = bottom terminal
    "res": [
        (16, 16),
        (16, 96),
    ],  # Verified: R1 at (-1728,-176) connects to (-1712,-160),(-1712,-80)
    "res2": [(16, 16), (16, 96)],
    "cap": [(16, 16), (16, 80)],  # Capacitor slightly shorter than resistor
    "cap2": [(16, 16), (16, 80)],
    "ind": [(16, 16), (16, 96)],  # Same as resistor
    "ind2": [(16, 16), (16, 96)],
    "voltage": [
        (0, 16),
        (0, 96),
    ],  # Verified: V1 at (-1248,-832) connects to (-1248,-816),(-1248,-736)
    "current": [(0, 16), (0, 96)],  # Same as voltage
    "diode": [(16, 16), (16, 80)],  # Same as capacitor
    "dio": [(16, 16), (16, 80)],
    # Op-amp (3 pins: inp, inn, out)
    # LTspice UniversalOpAmp pinout - verified from actual ASC files
    # Pins are relative to symbol center, rotation R0
    # IMPORTANT: In LTspice screen coords, Y increases downward, so:
    #   - Pin at (-32, -16) is ABOVE center = inverting input (inn, -)
    #   - Pin at (-32, 16) is BELOW center = non-inverting input (inp, +)
    #   - Pin at (32, 0) is RIGHT = output
    "opamp": [(-32, 16), (-32, -16), (32, 0)],  # inp (+), inn (-), out
    "opamp2": [(-32, 16), (-32, -16), (32, 0)],
    "universalopamp": [(-32, 16), (-32, -16), (32, 0)],  # Verified from PT1000_circuit_1.asc
    "universalopamp2": [(-32, 16), (-32, -16), (32, 0)],
    # 4-terminal controlled sources
    "e": [(16, 16), (16, 96), (32, 32), (32, 80)],  # p, n, cp, cn
    "g": [(16, 16), (16, 96), (32, 32), (32, 80)],
    "f": [(16, 16), (16, 96)],  # 2 terminal + ctrl ref
    "h": [(16, 16), (16, 96)],
}

# Default 2-terminal pin offsets (same as resistor)
DEFAULT_2PIN = [(16, 16), (16, 96)]


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
        if label:
            # Set label on the ROOT of the set, not just the point
            root = self.find(point)
            if root not in self._labels:
                self._labels[root] = label

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
        # Prefer labeled root - if either has a label, make it the root
        if ra in self._labels and rb not in self._labels:
            self._parent[rb] = ra
        elif rb in self._labels and ra not in self._labels:
            self._parent[ra] = rb
        elif ra in self._labels and rb in self._labels:
            # Both have labels - prefer ra, but don't lose rb's label
            # (This shouldn't happen in normal circuits)
            self._parent[rb] = ra
        else:
            # Neither has a label - just pick one
            self._parent[ra] = rb

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

    # Note: Analysis commands (.tran, .ac, .dc) are NOT added here because they
    # will be added by the simulation engine. Adding them here would cause
    # duplicate analysis commands which LTspice rejects.
    # Use get_analyses_from_asc() to extract analysis commands for simulation.

    # Add measurements (these are extracted by the simulator)
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


# ---------------------------------------------------------------------------
# Engineering Notation Parser
# ---------------------------------------------------------------------------

# Engineering notation suffixes
_ENG_SUFFIXES = {
    "f": 1e-15,
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "m": 1e-3,
    "k": 1e3,
    "K": 1e3,
    "meg": 1e6,
    "MEG": 1e6,
    "M": 1e6,
    "g": 1e9,
    "G": 1e9,
    "t": 1e12,
    "T": 1e12,
}


def parse_eng_number(value: str) -> float:
    """Parse a number with optional engineering suffix.

    Args:
        value: String like "5m", "10k", "1.5meg", "100n", "1e-3", etc.

    Returns:
        The numeric value as a float

    Example:
        >>> parse_eng_number("5m")
        0.005
        >>> parse_eng_number("10k")
        10000.0
        >>> parse_eng_number("1.5meg")
        1500000.0
    """
    value = value.strip()

    # Try direct float conversion first
    try:
        return float(value)
    except ValueError:
        pass

    # Try to extract number and suffix
    match = re.match(r"^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*(\w+)?$", value)
    if not match:
        raise ValueError(f"Cannot parse number: {value}")

    num_str, suffix = match.groups()
    num = float(num_str)

    if suffix:
        suffix_lower = suffix.lower()
        # Check for MEG first (case insensitive)
        if suffix_lower == "meg":
            return num * 1e6
        # Then check single character suffixes
        if suffix[0] in _ENG_SUFFIXES:
            return num * _ENG_SUFFIXES[suffix[0]]

    return num


# ---------------------------------------------------------------------------
# Analysis Command Parsing
# ---------------------------------------------------------------------------


def parse_tran_args(parameters: str) -> dict[str, float]:
    """Parse .tran parameters into AnalysisSpec args.

    LTspice .tran syntax: .tran <tstep> <tstop> [<tstart>] [<tmaxstep>] [<options>]

    Args:
        parameters: The parameter string after ".tran"

    Returns:
        Dictionary with tstep, tstop, and optionally tstart, tmax
    """
    parts = parameters.split()
    result: dict[str, float] = {}

    if len(parts) >= 1:
        result["tstep"] = parse_eng_number(parts[0])
    if len(parts) >= 2:
        result["tstop"] = parse_eng_number(parts[1])
    if len(parts) >= 3:
        result["tstart"] = parse_eng_number(parts[2])
    if len(parts) >= 4:
        result["tmax"] = parse_eng_number(parts[3])

    return result


def parse_ac_args(parameters: str) -> dict[str, Any]:
    """Parse .ac parameters into AnalysisSpec args.

    LTspice .ac syntax: .ac <type> <npoints> <fstart> <fstop>
    Types: DEC (decade), OCT (octave), LIN (linear)

    Args:
        parameters: The parameter string after ".ac"

    Returns:
        Dictionary with variation, npoints, fstart, fstop
    """
    parts = parameters.split()
    result: dict[str, Any] = {}

    if len(parts) >= 1:
        result["variation"] = parts[0].lower()
    if len(parts) >= 2:
        result["npoints"] = int(parse_eng_number(parts[1]))
    if len(parts) >= 3:
        result["fstart"] = parse_eng_number(parts[2])
    if len(parts) >= 4:
        result["fstop"] = parse_eng_number(parts[3])

    return result


def parse_dc_args(parameters: str) -> dict[str, Any]:
    """Parse .dc parameters into AnalysisSpec args.

    LTspice .dc syntax: .dc <source> <start> <stop> <step>

    Args:
        parameters: The parameter string after ".dc"

    Returns:
        Dictionary with src, start, stop, step
    """
    parts = parameters.split()
    result: dict[str, Any] = {}

    if len(parts) >= 1:
        result["src"] = parts[0]
    if len(parts) >= 2:
        result["start"] = parse_eng_number(parts[1])
    if len(parts) >= 3:
        result["stop"] = parse_eng_number(parts[2])
    if len(parts) >= 4:
        result["step"] = parse_eng_number(parts[3])

    return result


def parse_analysis_command(analysis_type: str, parameters: str) -> dict[str, Any] | None:
    """Parse an analysis command into mode and args.

    Args:
        analysis_type: The type of analysis (tran, ac, dc, op)
        parameters: The raw parameter string

    Returns:
        Dictionary with 'mode' and 'args' keys, or None if unsupported
    """
    analysis_type = analysis_type.lower()

    if analysis_type == "tran":
        return {"mode": "tran", "args": parse_tran_args(parameters)}
    elif analysis_type == "ac":
        return {"mode": "ac", "args": parse_ac_args(parameters)}
    elif analysis_type == "dc":
        return {"mode": "dc", "args": parse_dc_args(parameters)}
    elif analysis_type == "op":
        return {"mode": "op", "args": {}}
    else:
        return None


def get_analyses_from_asc(asc_result: AscParseResult) -> list[dict[str, Any]]:
    """Extract and parse all analysis commands from ASC parse result.

    Args:
        asc_result: The parsed ASC file result

    Returns:
        List of analysis specs as dictionaries with 'mode' and 'args'
    """
    analyses = []
    for cmd in asc_result.analysis_commands:
        parsed = parse_analysis_command(cmd.analysis_type, cmd.parameters)
        if parsed:
            analyses.append(parsed)
    return analyses


# ---------------------------------------------------------------------------
# High-Level Simulation Function
# ---------------------------------------------------------------------------


def run_asc_simulation(
    path: str,
    *,
    engine: str = "ltspice",
) -> Any:
    """Load an ASC file and run simulation using LTspice (default).

    This is the recommended way to simulate LTspice .asc files.
    It automatically:
    - Parses the ASC file
    - Converts to SpiceLab Circuit
    - Extracts analysis commands
    - Runs the simulation with LTspice

    Args:
        path: Path to the .asc file
        engine: Simulation engine ("ltspice" or "ngspice"). Default is "ltspice"
                since ASC files are LTspice native format.

    Returns:
        ResultHandle with simulation results

    Example:
        >>> result = run_asc_simulation("circuit.asc")
        >>> ds = result.dataset()
        >>> print(ds["V(vout)"].values)
    """
    from ..core.types import AnalysisSpec
    from ..engines import run_simulation
    from .asc_parser import parse_asc_file

    # Parse and convert
    asc_result = parse_asc_file(path)
    conv_result = asc_to_circuit(asc_result)

    if not conv_result.success:
        log.warning(
            "Some components were skipped during conversion: %s",
            conv_result.skipped_components,
        )

    # Get analyses from ASC file
    analyses_data = get_analyses_from_asc(asc_result)
    if not analyses_data:
        raise ValueError("No supported analysis commands found in ASC file")

    # Convert to AnalysisSpec objects
    analyses = [AnalysisSpec(mode=a["mode"], args=a["args"]) for a in analyses_data]

    # Run simulation with specified engine (default: ltspice)
    return run_simulation(
        conv_result.circuit,
        analyses,
        engine=engine,
    )


__all__ = [
    "ConversionWarning",
    "ConversionResult",
    "asc_to_circuit",
    "load_circuit_from_asc",
    "print_conversion_result",
    "SYMBOL_MAP",
    "KNOWN_UNSUPPORTED",
    # Analysis parsing
    "parse_eng_number",
    "parse_tran_args",
    "parse_ac_args",
    "parse_dc_args",
    "parse_analysis_command",
    "get_analyses_from_asc",
    # High-level simulation
    "run_asc_simulation",
]
