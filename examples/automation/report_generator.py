"""Design Report Generator

Demonstrates automated generation of design documentation
from circuit analysis results.

Run: python examples/automation/report_generator.py
"""

import math
from dataclasses import dataclass
from datetime import datetime

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net


@dataclass
class DesignInput:
    """Design input specification."""

    name: str
    value: float
    unit: str
    description: str


@dataclass
class DesignOutput:
    """Calculated design output."""

    name: str
    value: float
    unit: str
    spec_min: float | None = None
    spec_max: float | None = None

    @property
    def in_spec(self) -> bool | None:
        """Check if value is within specification."""
        if self.spec_min is None and self.spec_max is None:
            return None
        if self.spec_min is not None and self.value < self.spec_min:
            return False
        if self.spec_max is not None and self.value > self.spec_max:
            return False
        return True

    @property
    def status(self) -> str:
        """Get status string."""
        result = self.in_spec
        if result is None:
            return "INFO"
        return "PASS" if result else "FAIL"


class ReportGenerator:
    """Generate design reports."""

    def __init__(self, title: str, author: str = "SpiceLab"):
        self.title = title
        self.author = author
        self.date = datetime.now().strftime("%Y-%m-%d")
        self.inputs: list[DesignInput] = []
        self.outputs: list[DesignOutput] = []
        self.notes: list[str] = []
        self.warnings: list[str] = []

    def add_input(self, name: str, value: float, unit: str, description: str = ""):
        """Add a design input."""
        self.inputs.append(DesignInput(name, value, unit, description))

    def add_output(
        self,
        name: str,
        value: float,
        unit: str,
        spec_min: float | None = None,
        spec_max: float | None = None,
    ):
        """Add a calculated output."""
        self.outputs.append(DesignOutput(name, value, unit, spec_min, spec_max))

    def add_note(self, note: str):
        """Add a design note."""
        self.notes.append(note)

    def add_warning(self, warning: str):
        """Add a design warning."""
        self.warnings.append(warning)

    def format_value(self, value: float, unit: str) -> str:
        """Format value with engineering notation."""
        if value == 0:
            return f"0 {unit}"

        prefixes = [
            (1e12, "T"),
            (1e9, "G"),
            (1e6, "M"),
            (1e3, "k"),
            (1, ""),
            (1e-3, "m"),
            (1e-6, "µ"),
            (1e-9, "n"),
            (1e-12, "p"),
        ]

        for threshold, prefix in prefixes:
            if abs(value) >= threshold:
                return f"{value/threshold:.3g} {prefix}{unit}"

        return f"{value:.3g} {unit}"

    def generate_text_report(self) -> str:
        """Generate a plain text report."""
        lines = []
        width = 70

        # Header
        lines.append("=" * width)
        lines.append(f"  {self.title}")
        lines.append("=" * width)
        lines.append(f"  Author: {self.author}")
        lines.append(f"  Date: {self.date}")
        lines.append("")

        # Inputs
        lines.append("  DESIGN INPUTS")
        lines.append("  " + "-" * (width - 4))
        for inp in self.inputs:
            val_str = self.format_value(inp.value, inp.unit)
            lines.append(f"  {inp.name:20s}: {val_str:>15s}  {inp.description}")
        lines.append("")

        # Outputs
        lines.append("  CALCULATED OUTPUTS")
        lines.append("  " + "-" * (width - 4))
        for out in self.outputs:
            val_str = self.format_value(out.value, out.unit)
            status = out.status
            spec_str = ""
            if out.spec_min is not None or out.spec_max is not None:
                min_str = self.format_value(out.spec_min, out.unit) if out.spec_min else "-∞"
                max_str = self.format_value(out.spec_max, out.unit) if out.spec_max else "+∞"
                spec_str = f"[{min_str} to {max_str}]"
            lines.append(f"  {out.name:20s}: {val_str:>15s}  [{status:4s}] {spec_str}")
        lines.append("")

        # Summary
        pass_count = sum(1 for o in self.outputs if o.in_spec is True)
        fail_count = sum(1 for o in self.outputs if o.in_spec is False)
        info_count = sum(1 for o in self.outputs if o.in_spec is None)

        lines.append("  SUMMARY")
        lines.append("  " + "-" * (width - 4))
        lines.append(f"  Specifications: {pass_count} PASS, {fail_count} FAIL, {info_count} INFO")

        if fail_count > 0:
            lines.append("  Status: DESIGN DOES NOT MEET SPECIFICATIONS")
        else:
            lines.append("  Status: DESIGN MEETS ALL SPECIFICATIONS")
        lines.append("")

        # Warnings
        if self.warnings:
            lines.append("  WARNINGS")
            lines.append("  " + "-" * (width - 4))
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
            lines.append("")

        # Notes
        if self.notes:
            lines.append("  NOTES")
            lines.append("  " + "-" * (width - 4))
            for n in self.notes:
                lines.append(f"  • {n}")
            lines.append("")

        lines.append("=" * width)

        return "\n".join(lines)

    def generate_markdown_report(self) -> str:
        """Generate a Markdown report."""
        lines = []

        # Header
        lines.append(f"# {self.title}")
        lines.append("")
        lines.append(f"**Author:** {self.author}")
        lines.append(f"**Date:** {self.date}")
        lines.append("")

        # Inputs
        lines.append("## Design Inputs")
        lines.append("")
        lines.append("| Parameter | Value | Description |")
        lines.append("|-----------|-------|-------------|")
        for inp in self.inputs:
            val_str = self.format_value(inp.value, inp.unit)
            lines.append(f"| {inp.name} | {val_str} | {inp.description} |")
        lines.append("")

        # Outputs
        lines.append("## Calculated Outputs")
        lines.append("")
        lines.append("| Parameter | Value | Status | Specification |")
        lines.append("|-----------|-------|--------|---------------|")
        for out in self.outputs:
            val_str = self.format_value(out.value, out.unit)
            status = out.status
            spec_str = ""
            if out.spec_min is not None or out.spec_max is not None:
                min_str = self.format_value(out.spec_min, out.unit) if out.spec_min else "-∞"
                max_str = self.format_value(out.spec_max, out.unit) if out.spec_max else "+∞"
                spec_str = f"{min_str} to {max_str}"
            lines.append(f"| {out.name} | {val_str} | {status} | {spec_str} |")
        lines.append("")

        # Warnings
        if self.warnings:
            lines.append("## Warnings")
            lines.append("")
            for w in self.warnings:
                lines.append(f"- ⚠️ {w}")
            lines.append("")

        # Notes
        if self.notes:
            lines.append("## Notes")
            lines.append("")
            for n in self.notes:
                lines.append(f"- {n}")
            lines.append("")

        return "\n".join(lines)


def build_rc_filter(r: float, c: float) -> Circuit:
    """Build RC lowpass filter for analysis."""
    circuit = Circuit("rc_lowpass")

    v_in = Vdc("in", 5.0)
    r1 = Resistor("1", resistance=r)
    c1 = Capacitor("1", capacitance=c)

    circuit.add(v_in, r1, c1)

    vin = Net("vin")
    vout = Net("vout")

    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], vout)
    circuit.connect(c1.ports[0], vout)
    circuit.connect(c1.ports[1], GND)

    return circuit


def main():
    """Demonstrate report generation."""
    print("=" * 60)
    print("Automation: Design Report Generator")
    print("=" * 60)

    # Design parameters
    r_value = 10_000
    c_value = 100e-9
    vin = 5.0

    # Calculated values
    fc = 1 / (2 * math.pi * r_value * c_value)
    time_constant = r_value * c_value
    impedance_at_fc = r_value / math.sqrt(2)

    # Build circuit
    circuit = build_rc_filter(r_value, c_value)

    # Create report
    report = ReportGenerator(title="RC Lowpass Filter Design", author="SpiceLab Automation")

    # Add inputs
    report.add_input("Resistance", r_value, "Ω", "Series resistor")
    report.add_input("Capacitance", c_value, "F", "Shunt capacitor")
    report.add_input("Input Voltage", vin, "V", "DC bias")

    # Add outputs with specifications
    report.add_output("Cutoff Frequency", fc, "Hz", spec_min=100, spec_max=1000)
    report.add_output("Time Constant", time_constant, "s")
    report.add_output("Impedance at fc", impedance_at_fc, "Ω", spec_max=10000)
    report.add_output("-3dB Attenuation", -3.01, "dB")
    report.add_output("Phase at fc", -45, "deg")

    # Add notes
    report.add_note("First-order RC lowpass filter")
    report.add_note("Roll-off rate: -20 dB/decade above fc")
    report.add_note("Use NPO/C0G capacitors for stability")

    # Check for warnings
    if fc < 100:
        report.add_warning("Cutoff frequency very low, long settling time")
    if r_value > 100_000:
        report.add_warning("High resistance may introduce noise")

    # Validate circuit
    result = circuit.validate()
    if not result.is_valid:
        report.add_warning("Circuit validation failed")

    # Generate and display text report
    print("\n   TEXT REPORT OUTPUT:")
    print()
    print(report.generate_text_report())

    # Show Markdown preview
    print("\n   MARKDOWN REPORT PREVIEW (first 30 lines):")
    print("   " + "-" * 50)
    md_lines = report.generate_markdown_report().split("\n")[:30]
    for line in md_lines:
        print(f"   {line}")
    print("   ...")

    print("""
   Report Formats Available:
   ┌────────────────────────────────────────────────────────┐
   │ Text:     Plain text for console/logs                  │
   │ Markdown: GitHub/documentation compatible              │
   │ HTML:     Web-ready reports (extend class)             │
   │ PDF:      Print-ready documents (extend class)         │
   │ JSON:     Machine-readable for automation              │
   └────────────────────────────────────────────────────────┘

   Integration Ideas:
   - CI/CD: Auto-generate reports on design changes
   - Version control: Track spec compliance over time
   - Review: Attach reports to pull requests
   - Documentation: Include in project wiki
""")


if __name__ == "__main__":
    main()
