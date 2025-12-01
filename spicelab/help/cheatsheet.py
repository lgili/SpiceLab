"""API cheat sheet generator.

Generates quick reference documentation in various formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class CheatsheetFormat(Enum):
    """Output format for cheat sheet."""

    TEXT = auto()
    MARKDOWN = auto()
    HTML = auto()


@dataclass
class APIEntry:
    """An entry in the API cheat sheet."""

    name: str
    category: str
    signature: str
    description: str
    example: str = ""
    see_also: list[str] = field(default_factory=list)


@dataclass
class CheatsheetSection:
    """A section of the cheat sheet."""

    title: str
    description: str
    entries: list[APIEntry] = field(default_factory=list)


# Pre-defined API entries for the cheat sheet
CHEATSHEET_DATA = [
    CheatsheetSection(
        title="Circuit Creation",
        description="Creating and managing circuits",
        entries=[
            APIEntry(
                name="Circuit",
                category="core",
                signature="Circuit(name: str)",
                description="Create a new circuit container",
                example="circuit = Circuit('my_circuit')",
            ),
            APIEntry(
                name="circuit.add",
                category="core",
                signature="circuit.add(*components)",
                description="Add one or more components to circuit",
                example="circuit.add(R1, R2, C1)",
            ),
            APIEntry(
                name="circuit.connect",
                category="core",
                signature="circuit.connect(port, net)",
                description="Connect a component port to a net",
                example="circuit.connect(R1.ports[0], vin)",
            ),
            APIEntry(
                name="circuit.validate",
                category="core",
                signature="circuit.validate() -> ValidationResult",
                description="Check circuit for errors",
                example="result = circuit.validate()",
            ),
            APIEntry(
                name="circuit.build_netlist",
                category="core",
                signature="circuit.build_netlist() -> str",
                description="Generate SPICE netlist string",
                example="netlist = circuit.build_netlist()",
            ),
            APIEntry(
                name="circuit.preview_netlist",
                category="core",
                signature="circuit.preview_netlist() -> str",
                description="Formatted netlist with highlighting",
                example="print(circuit.preview_netlist())",
            ),
        ],
    ),
    CheatsheetSection(
        title="Components",
        description="Creating electronic components",
        entries=[
            APIEntry(
                name="Resistor",
                category="components",
                signature="Resistor(ref, resistance)",
                description="Create a resistor",
                example="R1 = Resistor(ref='1', resistance=10_000)",
            ),
            APIEntry(
                name="Capacitor",
                category="components",
                signature="Capacitor(ref, capacitance)",
                description="Create a capacitor",
                example="C1 = Capacitor(ref='1', capacitance=100e-9)",
            ),
            APIEntry(
                name="Inductor",
                category="components",
                signature="Inductor(ref, inductance)",
                description="Create an inductor",
                example="L1 = Inductor(ref='1', inductance=10e-3)",
            ),
            APIEntry(
                name="Vdc",
                category="components",
                signature="Vdc(ref, value)",
                description="DC voltage source",
                example="V1 = Vdc(ref='1', value='5')",
            ),
            APIEntry(
                name="Vac",
                category="components",
                signature="Vac(ref, value, ac_mag)",
                description="AC voltage source",
                example="V1 = Vac(ref='1', value='0', ac_mag='1')",
            ),
            APIEntry(
                name="Diode",
                category="components",
                signature="Diode(ref, model)",
                description="Diode with model",
                example="D1 = Diode(ref='1', model='1N4148')",
            ),
        ],
    ),
    CheatsheetSection(
        title="Nets and Connections",
        description="Electrical connections",
        entries=[
            APIEntry(
                name="Net",
                category="net",
                signature="Net(name: str)",
                description="Create a named electrical node",
                example="vin = Net('vin')",
            ),
            APIEntry(
                name="GND",
                category="net",
                signature="GND",
                description="Global ground reference (node 0)",
                example="circuit.connect(R1.ports[1], GND)",
            ),
        ],
    ),
    CheatsheetSection(
        title="Templates",
        description="Pre-built circuit templates",
        entries=[
            APIEntry(
                name="rc_lowpass",
                category="templates",
                signature="rc_lowpass(fc: float) -> Circuit",
                description="RC lowpass filter with cutoff frequency",
                example="circuit = rc_lowpass(fc=1000)",
            ),
            APIEntry(
                name="rc_highpass",
                category="templates",
                signature="rc_highpass(fc: float) -> Circuit",
                description="RC highpass filter with cutoff frequency",
                example="circuit = rc_highpass(fc=1000)",
            ),
            APIEntry(
                name="voltage_divider",
                category="templates",
                signature="voltage_divider(ratio: float) -> Circuit",
                description="Resistive voltage divider",
                example="circuit = voltage_divider(ratio=0.5)",
            ),
            APIEntry(
                name="inverting_amp",
                category="templates",
                signature="inverting_amp(gain: float) -> Circuit",
                description="Inverting op-amp amplifier",
                example="circuit = inverting_amp(gain=10)",
            ),
        ],
    ),
    CheatsheetSection(
        title="Simulation",
        description="Running simulations",
        entries=[
            APIEntry(
                name="quick_ac",
                category="simulation",
                signature="quick_ac(circuit, start, stop, points=20)",
                description="Run AC frequency sweep",
                example="result = quick_ac(circuit, start=1, stop=1e6)",
            ),
            APIEntry(
                name="quick_tran",
                category="simulation",
                signature="quick_tran(circuit, duration, step=None)",
                description="Run transient analysis",
                example="result = quick_tran(circuit, duration='1ms')",
            ),
            APIEntry(
                name="quick_op",
                category="simulation",
                signature="quick_op(circuit)",
                description="Calculate DC operating point",
                example="result = quick_op(circuit)",
            ),
        ],
    ),
    CheatsheetSection(
        title="Debugging",
        description="Debugging and validation tools",
        entries=[
            APIEntry(
                name="dry_run",
                category="debug",
                signature="dry_run(circuit, analyses) -> DryRunResult",
                description="Validate without simulating",
                example="result = dry_run(circuit, analyses)",
            ),
            APIEntry(
                name="VerboseSimulation",
                category="debug",
                signature="with VerboseSimulation():",
                description="Enable verbose logging",
                example="with VerboseSimulation():\n    result = quick_ac(...)",
            ),
            APIEntry(
                name="SimulationDebugger",
                category="debug",
                signature="SimulationDebugger(circuit, analyses)",
                description="Step-by-step simulation debugging",
                example="debugger = SimulationDebugger(circuit, analyses)",
            ),
        ],
    ),
    CheatsheetSection(
        title="UX Tools",
        description="User experience utilities",
        entries=[
            APIEntry(
                name="ProgressBar",
                category="ux",
                signature="ProgressBar(total, desc)",
                description="Progress bar with ETA",
                example="with ProgressBar(total=100) as pbar:\n    pbar.update(1)",
            ),
            APIEntry(
                name="CircuitHistory",
                category="ux",
                signature="CircuitHistory(circuit)",
                description="Undo/redo for circuits",
                example="history = CircuitHistory(circuit)\nhistory.undo()",
            ),
            APIEntry(
                name="diff_circuits",
                category="ux",
                signature="diff_circuits(c1, c2) -> CircuitDiff",
                description="Compare two circuits",
                example="diff = diff_circuits(v1, v2)",
            ),
            APIEntry(
                name="BookmarkManager",
                category="ux",
                signature="BookmarkManager()",
                description="Save/load circuit configurations",
                example="manager.save_circuit('name', circuit)",
            ),
        ],
    ),
]


def generate_cheatsheet(
    format: CheatsheetFormat = CheatsheetFormat.MARKDOWN,
    sections: list[str] | None = None,
) -> str:
    """Generate API cheat sheet.

    Args:
        format: Output format (TEXT, MARKDOWN, HTML)
        sections: List of section titles to include (None = all)

    Returns:
        Formatted cheat sheet string

    Example:
        >>> from spicelab.help import generate_cheatsheet, CheatsheetFormat
        >>> print(generate_cheatsheet(CheatsheetFormat.MARKDOWN))
        >>> print(generate_cheatsheet(CheatsheetFormat.HTML))
    """
    data = CHEATSHEET_DATA
    if sections:
        data = [s for s in data if s.title in sections]

    if format == CheatsheetFormat.TEXT:
        return _generate_text(data)
    elif format == CheatsheetFormat.MARKDOWN:
        return _generate_markdown(data)
    elif format == CheatsheetFormat.HTML:
        return _generate_html(data)
    else:
        return _generate_text(data)


def _generate_text(sections: list[CheatsheetSection]) -> str:
    """Generate plain text cheat sheet."""
    lines = [
        "=" * 70,
        "SpiceLab API Cheat Sheet",
        "=" * 70,
        "",
    ]

    for section in sections:
        lines.append(f"\n{section.title}")
        lines.append("-" * len(section.title))
        lines.append(section.description)
        lines.append("")

        for entry in section.entries:
            lines.append(f"  {entry.name}")
            lines.append(f"    {entry.signature}")
            lines.append(f"    {entry.description}")
            if entry.example:
                lines.append(f"    Example: {entry.example.split(chr(10))[0]}")
            lines.append("")

    return "\n".join(lines)


def _generate_markdown(sections: list[CheatsheetSection]) -> str:
    """Generate Markdown cheat sheet."""
    lines = [
        "# SpiceLab API Cheat Sheet",
        "",
        "Quick reference for common SpiceLab operations.",
        "",
        "## Table of Contents",
        "",
    ]

    # TOC
    for section in sections:
        anchor = section.title.lower().replace(" ", "-")
        lines.append(f"- [{section.title}](#{anchor})")

    lines.append("")

    # Sections
    for section in sections:
        lines.append(f"## {section.title}")
        lines.append("")
        lines.append(section.description)
        lines.append("")
        lines.append("| Function | Description |")
        lines.append("|----------|-------------|")

        for entry in section.entries:
            desc = entry.description.replace("|", "\\|")
            lines.append(f"| `{entry.name}` | {desc} |")

        lines.append("")

        # Detailed entries
        for entry in section.entries:
            lines.append(f"### `{entry.name}`")
            lines.append("")
            lines.append("```python")
            lines.append(f"{entry.signature}")
            lines.append("```")
            lines.append("")
            lines.append(entry.description)
            lines.append("")

            if entry.example:
                lines.append("**Example:**")
                lines.append("```python")
                lines.append(entry.example)
                lines.append("```")
                lines.append("")

    return "\n".join(lines)


def _generate_html(sections: list[CheatsheetSection]) -> str:
    """Generate HTML cheat sheet."""
    css = """
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 900px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #2980b9; margin-top: 30px; }
        h3 { color: #27ae60; }
        .section { margin-bottom: 30px; }
        .entry { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px;
                 border-left: 4px solid #3498db; }
        .signature { font-family: 'Monaco', 'Menlo', monospace; background: #2c3e50;
                     color: #ecf0f1; padding: 8px 12px; border-radius: 3px; display: inline-block; }
        .description { margin: 10px 0; color: #555; }
        .example { background: #1e1e1e; color: #d4d4d4; padding: 10px; border-radius: 3px;
                   font-family: monospace; overflow-x: auto; white-space: pre; }
        .toc { background: #ecf0f1; padding: 15px; border-radius: 5px; }
        .toc ul { list-style: none; padding-left: 0; }
        .toc li { margin: 5px 0; }
        .toc a { color: #2980b9; text-decoration: none; }
        .toc a:hover { text-decoration: underline; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #3498db; color: white; }
        tr:hover { background: #f5f5f5; }
        code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-family: monospace; }
    </style>
    """

    lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='UTF-8'>",
        "<title>SpiceLab API Cheat Sheet</title>",
        css,
        "</head>",
        "<body>",
        "<h1>SpiceLab API Cheat Sheet</h1>",
        "<p>Quick reference for common SpiceLab operations.</p>",
        "",
        "<div class='toc'>",
        "<strong>Table of Contents</strong>",
        "<ul>",
    ]

    # TOC
    for section in sections:
        anchor = section.title.lower().replace(" ", "-")
        lines.append(f"<li><a href='#{anchor}'>{section.title}</a></li>")

    lines.append("</ul>")
    lines.append("</div>")

    # Sections
    for section in sections:
        anchor = section.title.lower().replace(" ", "-")
        lines.append("<div class='section'>")
        lines.append(f"<h2 id='{anchor}'>{section.title}</h2>")
        lines.append(f"<p>{section.description}</p>")

        # Summary table
        lines.append("<table>")
        lines.append("<tr><th>Function</th><th>Description</th></tr>")
        for entry in section.entries:
            lines.append(f"<tr><td><code>{entry.name}</code></td><td>{entry.description}</td></tr>")
        lines.append("</table>")

        # Detailed entries
        for entry in section.entries:
            lines.append("<div class='entry'>")
            lines.append(f"<h3>{entry.name}</h3>")
            lines.append(f"<div class='signature'>{entry.signature}</div>")
            lines.append(f"<p class='description'>{entry.description}</p>")

            if entry.example:
                example_html = entry.example.replace("<", "&lt;").replace(">", "&gt;")
                lines.append("<strong>Example:</strong>")
                lines.append(f"<div class='example'>{example_html}</div>")

            lines.append("</div>")

        lines.append("</div>")

    lines.extend(
        [
            "</body>",
            "</html>",
        ]
    )

    return "\n".join(lines)


def save_cheatsheet(
    path: str,
    format: CheatsheetFormat | None = None,
) -> None:
    """Save cheat sheet to a file.

    Args:
        path: Output file path
        format: Format (auto-detected from extension if None)

    Example:
        >>> from spicelab.help import save_cheatsheet
        >>> save_cheatsheet("cheatsheet.md")
        >>> save_cheatsheet("cheatsheet.html")
    """
    from pathlib import Path

    p = Path(path)

    if format is None:
        ext = p.suffix.lower()
        if ext == ".html" or ext == ".htm":
            format = CheatsheetFormat.HTML
        elif ext == ".md":
            format = CheatsheetFormat.MARKDOWN
        else:
            format = CheatsheetFormat.TEXT

    content = generate_cheatsheet(format)
    p.write_text(content, encoding="utf-8")


__all__ = [
    "CheatsheetFormat",
    "APIEntry",
    "CheatsheetSection",
    "generate_cheatsheet",
    "save_cheatsheet",
    "CHEATSHEET_DATA",
]
