# mypy: ignore-errors
"""Comprehensive LTspice ``.asc`` (schematic) parser.

This module provides a complete parser for LTspice .asc files that extracts
ALL elements without requiring a predefined symbol catalog. Unlike the
`ltspice_asc` module which focuses on Circuit conversion, this parser
preserves all raw data from the schematic file.

Supported Elements
------------------
* Version - File format version
* SHEET - Sheet dimensions
* WIRE - Wire connections (copper traces)
* FLAG - Net labels/flags
* SYMBOL - Component symbols
* SYMATTR - Symbol attributes (InstName, Value, SpiceLine, etc.)
* WINDOW - Window positioning for attributes
* TEXT - Text annotations (SPICE directives and comments)
* LINE - Drawing lines
* RECTANGLE - Drawing rectangles
* CIRCLE - Drawing circles
* ARC - Drawing arcs
* BUSTAP - Bus taps
netlist-level data:
* .param - Parameters
* .meas - Measurements
* .tran/.ac/.dc/.op - Analysis commands

Example
-------
>>> from spicelab.io.asc_parser import parse_asc_file, AscParser
>>> result = parse_asc_file("circuit.asc")
>>> print(f"Components: {len(result.components)}")
>>> print(f"Wires: {len(result.wires)}")
>>> print(f"Parameters: {result.parameters}")
>>> for comp in result.components:
...     print(f"  {comp.ref}: {comp.symbol} = {comp.value}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data Classes for ASC Elements
# ---------------------------------------------------------------------------


@dataclass
class Point:
    """2D point with integer coordinates."""

    x: int
    y: int

    def __iter__(self):
        yield self.x
        yield self.y


@dataclass
class Wire:
    """Wire connection between two points."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def start(self) -> Point:
        return Point(self.x1, self.y1)

    @property
    def end(self) -> Point:
        return Point(self.x2, self.y2)

    @property
    def is_horizontal(self) -> bool:
        return self.y1 == self.y2

    @property
    def is_vertical(self) -> bool:
        return self.x1 == self.x2

    @property
    def length(self) -> float:
        return ((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2) ** 0.5


@dataclass
class Flag:
    """Net label/flag at a specific position."""

    name: str
    x: int
    y: int

    @property
    def position(self) -> Point:
        return Point(self.x, self.y)

    @property
    def is_ground(self) -> bool:
        return self.name in ("0", "GND", "gnd")


@dataclass
class Window:
    """Window positioning for symbol attributes."""

    index: int
    x: int
    y: int
    alignment: str
    font_size: int | None = None


@dataclass
class SymbolComponent:
    """Component instance from a SYMBOL definition."""

    symbol: str
    x: int
    y: int
    rotation: str
    attributes: dict[str, str] = field(default_factory=dict)
    windows: list[Window] = field(default_factory=list)

    @property
    def position(self) -> Point:
        return Point(self.x, self.y)

    @property
    def ref(self) -> str | None:
        """Reference designator (InstName)."""
        return self.attributes.get("InstName")

    @property
    def value(self) -> str | None:
        """Component value."""
        return self.attributes.get("Value")

    @property
    def model(self) -> str | None:
        """SPICE model name."""
        return self.attributes.get("SpiceModel")

    @property
    def spice_line(self) -> str | None:
        """SpiceLine attribute."""
        return self.attributes.get("SpiceLine")

    @property
    def spice_line2(self) -> str | None:
        """SpiceLine2 attribute."""
        return self.attributes.get("SpiceLine2")

    @property
    def symbol_basename(self) -> str:
        """Symbol name without path (e.g., 'UniversalOpAmp' from 'OpAmps\\UniversalOpAmp')."""
        return self.symbol.replace("\\", "/").split("/")[-1]

    @property
    def component_type(self) -> str:
        """Infer component type from reference designator."""
        ref = self.ref
        if not ref:
            return "unknown"
        prefix = ref.rstrip("0123456789")
        type_map = {
            "R": "resistor",
            "C": "capacitor",
            "L": "inductor",
            "V": "voltage_source",
            "I": "current_source",
            "D": "diode",
            "Q": "bjt",
            "M": "mosfet",
            "J": "jfet",
            "U": "ic",
            "X": "subcircuit",
        }
        return type_map.get(prefix, "unknown")


@dataclass
class TextElement:
    """Text annotation (SPICE directive or comment)."""

    x: int
    y: int
    alignment: str
    font_size: int
    text: str
    is_directive: bool = False

    @property
    def position(self) -> Point:
        return Point(self.x, self.y)

    @property
    def is_comment(self) -> bool:
        """Check if text is a comment (starts with ';')."""
        return self.text.startswith(";")

    @property
    def directive_type(self) -> str | None:
        """Extract directive type (e.g., 'tran', 'param', 'meas')."""
        if not self.is_directive:
            return None
        # Remove leading ! or . and get first word
        clean = self.text.lstrip("!.")
        match = re.match(r"(\w+)", clean)
        return match.group(1).lower() if match else None


@dataclass
class Line:
    """Drawing line element."""

    style: str
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class Rectangle:
    """Drawing rectangle element."""

    style: str
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class Circle:
    """Drawing circle element."""

    style: str
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class Arc:
    """Drawing arc element."""

    style: str
    x1: int
    y1: int
    x2: int
    y2: int
    x3: int
    y3: int
    x4: int
    y4: int


@dataclass
class BusTap:
    """Bus tap element."""

    x: int
    y: int


@dataclass
class Parameter:
    """SPICE parameter definition."""

    name: str
    value: str
    expression: str | None = None


@dataclass
class Measurement:
    """SPICE measurement definition."""

    name: str
    measurement_type: str
    expression: str


@dataclass
class AnalysisCommand:
    """SPICE analysis command."""

    analysis_type: str
    parameters: str
    raw_text: str


@dataclass
class AscParseResult:
    """Complete parse result from an ASC file."""

    # File metadata
    version: str
    sheet_number: int
    sheet_width: int
    sheet_height: int
    file_path: str | None = None

    # Electrical elements
    components: list[SymbolComponent] = field(default_factory=list)
    wires: list[Wire] = field(default_factory=list)
    flags: list[Flag] = field(default_factory=list)

    # Text elements
    texts: list[TextElement] = field(default_factory=list)

    # Drawing elements
    lines: list[Line] = field(default_factory=list)
    rectangles: list[Rectangle] = field(default_factory=list)
    circles: list[Circle] = field(default_factory=list)
    arcs: list[Arc] = field(default_factory=list)
    bus_taps: list[BusTap] = field(default_factory=list)

    # Extracted SPICE data
    parameters: dict[str, Parameter] = field(default_factory=dict)
    measurements: list[Measurement] = field(default_factory=list)
    analysis_commands: list[AnalysisCommand] = field(default_factory=list)

    # Raw data
    raw_lines: list[str] = field(default_factory=list)
    unknown_lines: list[str] = field(default_factory=list)

    @property
    def net_names(self) -> list[str]:
        """List of all net names from flags."""
        return [f.name for f in self.flags]

    @property
    def ground_flags(self) -> list[Flag]:
        """List of ground flags."""
        return [f for f in self.flags if f.is_ground]

    @property
    def directives(self) -> list[TextElement]:
        """List of SPICE directives."""
        return [t for t in self.texts if t.is_directive]

    @property
    def comments(self) -> list[TextElement]:
        """List of comment texts."""
        return [t for t in self.texts if t.is_comment]

    def get_components_by_type(self, prefix: str) -> list[SymbolComponent]:
        """Get components by reference prefix (e.g., 'R' for resistors)."""
        return [c for c in self.components if c.ref and c.ref.startswith(prefix)]

    def get_component_by_ref(self, ref: str) -> SymbolComponent | None:
        """Get a component by its reference designator."""
        for c in self.components:
            if c.ref == ref:
                return c
        return None

    def summary(self) -> dict[str, Any]:
        """Return a summary of the parsed file."""
        return {
            "version": self.version,
            "sheet_size": f"{self.sheet_width}x{self.sheet_height}",
            "components": len(self.components),
            "wires": len(self.wires),
            "flags": len(self.flags),
            "parameters": len(self.parameters),
            "measurements": len(self.measurements),
            "analysis_commands": len(self.analysis_commands),
            "texts": len(self.texts),
            "unknown_lines": len(self.unknown_lines),
        }


# ---------------------------------------------------------------------------
# Parser Class
# ---------------------------------------------------------------------------


class AscParser:
    """Parser for LTspice .asc schematic files.

    This parser extracts all elements from an ASC file without requiring
    a predefined symbol catalog. It preserves the raw data structure.

    Example
    -------
    >>> parser = AscParser()
    >>> result = parser.parse_file("circuit.asc")
    >>> print(result.summary())
    """

    # Regex patterns for parsing
    TEXT_PATTERN = re.compile(r"TEXT\s+(-?\d+)\s+(-?\d+)\s+(\w+)\s+(\d+)\s+(.*)")
    PARAM_PATTERN = re.compile(r"\.param\s+(\w+)\s*=\s*(.+)", re.IGNORECASE)
    MEAS_PATTERN = re.compile(r"\.meas\s+(\w+)\s+(\w+)\s+(.+)", re.IGNORECASE)
    ANALYSIS_PATTERN = re.compile(r"\.(tran|ac|dc|op|noise|tf|sens|four|pz)\s*(.*)", re.IGNORECASE)

    def __init__(self, encoding: str = "utf-8"):
        """Initialize parser.

        Parameters
        ----------
        encoding : str
            File encoding (default: utf-8)
        """
        self.encoding = encoding

    def parse_file(self, path: str | Path) -> AscParseResult:
        """Parse an ASC file.

        Parameters
        ----------
        path : str or Path
            Path to the .asc file

        Returns
        -------
        AscParseResult
            Complete parse result with all elements
        """
        path = Path(path)
        content = path.read_text(encoding=self.encoding, errors="ignore")
        result = self.parse_string(content)
        result.file_path = str(path)
        return result

    def parse_string(self, content: str) -> AscParseResult:
        """Parse ASC content from a string.

        Parameters
        ----------
        content : str
            ASC file content

        Returns
        -------
        AscParseResult
            Complete parse result with all elements
        """
        result = AscParseResult(
            version="4",
            sheet_number=1,
            sheet_width=880,
            sheet_height=680,
        )

        lines = content.splitlines()
        result.raw_lines = lines

        current_component: SymbolComponent | None = None

        for line in lines:
            line = line.rstrip()
            if not line:
                continue

            tokens = line.split()
            if not tokens:
                continue

            keyword = tokens[0]

            # Handle each line type
            if keyword == "Version" and len(tokens) >= 2:
                result.version = tokens[1]
                current_component = None

            elif keyword == "SHEET" and len(tokens) >= 4:
                result.sheet_number = int(tokens[1])
                result.sheet_width = int(tokens[2])
                result.sheet_height = int(tokens[3])
                current_component = None

            elif keyword == "WIRE" and len(tokens) >= 5:
                result.wires.append(
                    Wire(
                        x1=int(tokens[1]),
                        y1=int(tokens[2]),
                        x2=int(tokens[3]),
                        y2=int(tokens[4]),
                    )
                )
                current_component = None

            elif keyword == "FLAG" and len(tokens) >= 4:
                result.flags.append(
                    Flag(
                        name=tokens[3],
                        x=int(tokens[1]),
                        y=int(tokens[2]),
                    )
                )
                current_component = None

            elif keyword == "SYMBOL" and len(tokens) >= 5:
                comp = SymbolComponent(
                    symbol=tokens[1],
                    x=int(tokens[2]),
                    y=int(tokens[3]),
                    rotation=tokens[4],
                )
                result.components.append(comp)
                current_component = comp

            elif keyword == "SYMATTR" and len(tokens) >= 3 and current_component:
                attr_name = tokens[1]
                attr_value = " ".join(tokens[2:])
                current_component.attributes[attr_name] = attr_value

            elif keyword == "WINDOW" and len(tokens) >= 5 and current_component:
                window = Window(
                    index=int(tokens[1]),
                    x=int(tokens[2]),
                    y=int(tokens[3]),
                    alignment=tokens[4],
                    font_size=int(tokens[5]) if len(tokens) > 5 else None,
                )
                current_component.windows.append(window)

            elif keyword == "TEXT":
                self._parse_text(line, result)
                current_component = None

            elif keyword == "LINE" and len(tokens) >= 6:
                result.lines.append(
                    Line(
                        style=tokens[1],
                        x1=int(tokens[2]),
                        y1=int(tokens[3]),
                        x2=int(tokens[4]),
                        y2=int(tokens[5]),
                    )
                )
                current_component = None

            elif keyword == "RECTANGLE" and len(tokens) >= 6:
                result.rectangles.append(
                    Rectangle(
                        style=tokens[1],
                        x1=int(tokens[2]),
                        y1=int(tokens[3]),
                        x2=int(tokens[4]),
                        y2=int(tokens[5]),
                    )
                )
                current_component = None

            elif keyword == "CIRCLE" and len(tokens) >= 6:
                result.circles.append(
                    Circle(
                        style=tokens[1],
                        x1=int(tokens[2]),
                        y1=int(tokens[3]),
                        x2=int(tokens[4]),
                        y2=int(tokens[5]),
                    )
                )
                current_component = None

            elif keyword == "ARC" and len(tokens) >= 10:
                result.arcs.append(
                    Arc(
                        style=tokens[1],
                        x1=int(tokens[2]),
                        y1=int(tokens[3]),
                        x2=int(tokens[4]),
                        y2=int(tokens[5]),
                        x3=int(tokens[6]),
                        y3=int(tokens[7]),
                        x4=int(tokens[8]),
                        y4=int(tokens[9]),
                    )
                )
                current_component = None

            elif keyword == "BUSTAP" and len(tokens) >= 3:
                result.bus_taps.append(
                    BusTap(
                        x=int(tokens[1]),
                        y=int(tokens[2]),
                    )
                )
                current_component = None

            else:
                # Unknown line type
                result.unknown_lines.append(line)
                current_component = None

        # Post-process: extract parameters and measurements from directives
        self._extract_spice_data(result)

        return result

    def _parse_text(self, line: str, result: AscParseResult) -> None:
        """Parse a TEXT line."""
        match = self.TEXT_PATTERN.match(line)
        if not match:
            result.unknown_lines.append(line)
            return

        x = int(match.group(1))
        y = int(match.group(2))
        alignment = match.group(3)
        font_size = int(match.group(4))
        text = match.group(5)

        # Check if it's a directive (starts with !)
        is_directive = text.startswith("!")
        if is_directive:
            text = text[1:]  # Remove leading !

        result.texts.append(
            TextElement(
                x=x,
                y=y,
                alignment=alignment,
                font_size=font_size,
                text=text,
                is_directive=is_directive,
            )
        )

    def _extract_spice_data(self, result: AscParseResult) -> None:
        """Extract parameters, measurements, and analysis commands from directives."""
        for text in result.texts:
            if not text.is_directive:
                continue

            content = text.text.strip()

            # Check for .param
            param_match = self.PARAM_PATTERN.match(content)
            if param_match:
                name = param_match.group(1)
                value = param_match.group(2).strip()
                result.parameters[name] = Parameter(
                    name=name,
                    value=value,
                    expression=value if any(c in value for c in "+-*/()") else None,
                )
                continue

            # Check for .meas
            meas_match = self.MEAS_PATTERN.match(content)
            if meas_match:
                result.measurements.append(
                    Measurement(
                        name=meas_match.group(1),
                        measurement_type=meas_match.group(2),
                        expression=meas_match.group(3).strip(),
                    )
                )
                continue

            # Check for analysis commands
            analysis_match = self.ANALYSIS_PATTERN.match(content)
            if analysis_match:
                result.analysis_commands.append(
                    AnalysisCommand(
                        analysis_type=analysis_match.group(1).lower(),
                        parameters=analysis_match.group(2).strip(),
                        raw_text=content,
                    )
                )


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def parse_asc_file(path: str | Path) -> AscParseResult:
    """Parse an LTspice .asc file.

    Parameters
    ----------
    path : str or Path
        Path to the .asc file

    Returns
    -------
    AscParseResult
        Complete parse result with all elements

    Example
    -------
    >>> result = parse_asc_file("circuit.asc")
    >>> print(f"Found {len(result.components)} components")
    """
    parser = AscParser()
    return parser.parse_file(path)


def parse_asc_string(content: str) -> AscParseResult:
    """Parse LTspice .asc content from a string.

    Parameters
    ----------
    content : str
        ASC file content

    Returns
    -------
    AscParseResult
        Complete parse result with all elements
    """
    parser = AscParser()
    return parser.parse_string(content)


def print_asc_summary(result: AscParseResult) -> None:
    """Print a summary of the parsed ASC file.

    Parameters
    ----------
    result : AscParseResult
        Parse result to summarize
    """
    print("ASC File Summary")
    print("================")
    if result.file_path:
        print(f"File: {result.file_path}")
    print(f"Version: {result.version}")
    print(f"Sheet: {result.sheet_width}x{result.sheet_height}")
    print()

    print("Elements:")
    print(f"  Components: {len(result.components)}")
    print(f"  Wires: {len(result.wires)}")
    print(f"  Flags/Nets: {len(result.flags)}")
    print(f"  Text elements: {len(result.texts)}")
    print()

    if result.components:
        print("Components:")
        for comp in result.components:
            ref = comp.ref or "?"
            value = comp.value or comp.model or "-"
            print(f"  {ref}: {comp.symbol} = {value}")
        print()

    if result.flags:
        print("Net Names:")
        for flag in result.flags:
            marker = " (GND)" if flag.is_ground else ""
            print(f"  {flag.name}{marker} at ({flag.x}, {flag.y})")
        print()

    if result.parameters:
        print("Parameters:")
        for name, param in result.parameters.items():
            print(f"  {name} = {param.value}")
        print()

    if result.measurements:
        print("Measurements:")
        for meas in result.measurements:
            print(f"  {meas.name}: {meas.measurement_type} {meas.expression}")
        print()

    if result.analysis_commands:
        print("Analysis Commands:")
        for cmd in result.analysis_commands:
            print(f"  .{cmd.analysis_type} {cmd.parameters}")
        print()

    if result.unknown_lines:
        print(f"Unknown/Unhandled Lines: {len(result.unknown_lines)}")


__all__ = [
    # Data classes
    "Point",
    "Wire",
    "Flag",
    "Window",
    "SymbolComponent",
    "TextElement",
    "Line",
    "Rectangle",
    "Circle",
    "Arc",
    "BusTap",
    "Parameter",
    "Measurement",
    "AnalysisCommand",
    "AscParseResult",
    # Parser
    "AscParser",
    # Convenience functions
    "parse_asc_file",
    "parse_asc_string",
    "print_asc_summary",
]
