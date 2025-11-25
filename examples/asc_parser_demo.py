#!/usr/bin/env python3
"""Demonstration of the SpiceLab ASC Parser.

This example shows how to use the comprehensive ASC parser to load and
analyze LTspice schematic files (.asc). The parser extracts ALL elements
from the schematic without requiring a predefined symbol catalog.

Features demonstrated:
- Loading .asc files
- Extracting components, wires, nets, and parameters
- Analyzing SPICE directives (parameters, measurements, analysis commands)
- Working with component attributes
- Comparing multiple schematics
"""

from pathlib import Path

from spicelab.io import (
    parse_asc_file,
    print_asc_summary,
)


def demo_basic_parsing():
    """Demonstrate basic ASC file parsing."""
    print("=" * 70)
    print("DEMO 1: Basic ASC File Parsing")
    print("=" * 70)

    # Find the old sim files directory
    project_root = Path(__file__).parent.parent
    asc_files = list(project_root.glob("old/sim_files/*.asc"))

    if not asc_files:
        print("No .asc files found in old/sim_files/")
        return

    # Parse the first file
    asc_file = asc_files[0]
    print(f"\nParsing: {asc_file.name}")
    print("-" * 50)

    result = parse_asc_file(asc_file)

    # Print full summary
    print_asc_summary(result)


def demo_component_analysis():
    """Demonstrate component extraction and analysis."""
    print("\n" + "=" * 70)
    print("DEMO 2: Component Analysis")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    asc_file = project_root / "old/sim_files/PT1000_circuit_1.asc"

    if not asc_file.exists():
        print(f"File not found: {asc_file}")
        return

    result = parse_asc_file(asc_file)

    print(f"\nComponents in {asc_file.name}:")
    print("-" * 50)

    # Group components by type
    resistors = result.get_components_by_type("R")
    voltage_sources = result.get_components_by_type("V")
    opamps = result.get_components_by_type("U")

    print(f"\nResistors ({len(resistors)}):")
    for r in resistors:
        value = r.value or "?"
        print(f"  {r.ref}: {value} ohms")

    print(f"\nVoltage Sources ({len(voltage_sources)}):")
    for v in voltage_sources:
        value = v.value or "?"
        print(f"  {v.ref}: {value} V")

    print(f"\nOp-Amps ({len(opamps)}):")
    for u in opamps:
        symbol = u.symbol_basename
        print(f"  {u.ref}: {symbol}")

    # Show all attributes for a component
    if resistors:
        r = resistors[0]
        print(f"\nAll attributes for {r.ref}:")
        for key, value in r.attributes.items():
            print(f"  {key}: {value}")


def demo_parameter_extraction():
    """Demonstrate parameter and measurement extraction."""
    print("\n" + "=" * 70)
    print("DEMO 3: Parameter and Measurement Extraction")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    asc_file = project_root / "old/sim_files/PT1000_circuit_1.asc"

    if not asc_file.exists():
        print(f"File not found: {asc_file}")
        return

    result = parse_asc_file(asc_file)

    print(f"\nParameters in {asc_file.name}:")
    print("-" * 50)
    for name, param in result.parameters.items():
        is_expr = " (expression)" if param.expression else ""
        print(f"  {name} = {param.value}{is_expr}")

    print(f"\nMeasurements ({len(result.measurements)}):")
    print("-" * 50)
    for meas in result.measurements:
        print(f"  .meas {meas.name} {meas.measurement_type} {meas.expression}")

    print(f"\nAnalysis Commands ({len(result.analysis_commands)}):")
    print("-" * 50)
    for cmd in result.analysis_commands:
        print(f"  .{cmd.analysis_type} {cmd.parameters}")


def demo_net_analysis():
    """Demonstrate net/flag analysis."""
    print("\n" + "=" * 70)
    print("DEMO 4: Net/Flag Analysis")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    asc_file = project_root / "old/sim_files/PT1000_circuit_1.asc"

    if not asc_file.exists():
        print(f"File not found: {asc_file}")
        return

    result = parse_asc_file(asc_file)

    print(f"\nNets in {asc_file.name}:")
    print("-" * 50)

    # Separate ground and signal nets
    ground_nets = [f for f in result.flags if f.is_ground]
    signal_nets = [f for f in result.flags if not f.is_ground]

    print(f"\nGround connections ({len(ground_nets)}):")
    for f in ground_nets:
        print(f"  {f.name} at ({f.x}, {f.y})")

    print(f"\nSignal nets ({len(signal_nets)}):")
    for f in signal_nets:
        print(f"  {f.name} at ({f.x}, {f.y})")


def demo_wire_analysis():
    """Demonstrate wire analysis."""
    print("\n" + "=" * 70)
    print("DEMO 5: Wire Analysis")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    asc_file = project_root / "old/sim_files/PT1000_circuit_1.asc"

    if not asc_file.exists():
        print(f"File not found: {asc_file}")
        return

    result = parse_asc_file(asc_file)

    print(f"\nWire statistics for {asc_file.name}:")
    print("-" * 50)
    print(f"  Total wires: {len(result.wires)}")

    horizontal = sum(1 for w in result.wires if w.is_horizontal)
    vertical = sum(1 for w in result.wires if w.is_vertical)
    diagonal = len(result.wires) - horizontal - vertical

    print(f"  Horizontal: {horizontal}")
    print(f"  Vertical: {vertical}")
    print(f"  Diagonal: {diagonal}")

    if result.wires:
        total_length = sum(w.length for w in result.wires)
        print(f"  Total wire length: {total_length:.0f} units")


def demo_compare_schematics():
    """Demonstrate comparing multiple schematics."""
    print("\n" + "=" * 70)
    print("DEMO 6: Schematic Comparison")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    asc_files = sorted(project_root.glob("old/sim_files/*.asc"))

    if len(asc_files) < 2:
        print("Need at least 2 .asc files for comparison")
        return

    print("\nComparing PT1000 circuit variants:")
    print("-" * 50)
    print(f"{'File':<25} {'Components':>12} {'Wires':>8} {'Params':>8}")
    print("-" * 50)

    for asc_file in asc_files:
        result = parse_asc_file(asc_file)
        print(
            f"{asc_file.name:<25} "
            f"{len(result.components):>12} "
            f"{len(result.wires):>8} "
            f"{len(result.parameters):>8}"
        )

    # Detailed comparison of resistor values
    print("\n\nResistor values across variants:")
    print("-" * 70)

    # Collect all resistor refs
    all_refs = set()
    results = {}
    for asc_file in asc_files:
        result = parse_asc_file(asc_file)
        results[asc_file.name] = result
        for r in result.get_components_by_type("R"):
            if r.ref:
                all_refs.add(r.ref)

    # Print header
    header = f"{'Ref':<6}"
    for name in results:
        header += f" {name.replace('PT1000_circuit_', 'v'):<12}"
    print(header)
    print("-" * len(header))

    # Print values
    for ref in sorted(all_refs):
        row = f"{ref:<6}"
        for result in results.values():
            comp = result.get_component_by_ref(ref)
            value = comp.value if comp else "-"
            row += f" {str(value):<12}"
        print(row)


def demo_raw_data_access():
    """Demonstrate raw data access."""
    print("\n" + "=" * 70)
    print("DEMO 7: Raw Data Access")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    asc_file = project_root / "old/sim_files/PT1000_circuit_1.asc"

    if not asc_file.exists():
        print(f"File not found: {asc_file}")
        return

    result = parse_asc_file(asc_file)

    print(f"\nRaw data access for {asc_file.name}:")
    print("-" * 50)

    # Show file metadata
    print("\nFile metadata:")
    print(f"  Version: {result.version}")
    print(f"  Sheet size: {result.sheet_width}x{result.sheet_height}")

    # Show directive texts
    directives = result.directives
    print(f"\nSPICE Directives ({len(directives)}):")
    for d in directives[:5]:  # First 5
        print(f"  {d.text[:60]}...")

    # Show unknown/unhandled lines
    if result.unknown_lines:
        print(f"\nUnhandled lines ({len(result.unknown_lines)}):")
        for line in result.unknown_lines[:3]:
            print(f"  {line[:60]}...")


def demo_programmatic_usage():
    """Demonstrate programmatic usage patterns."""
    print("\n" + "=" * 70)
    print("DEMO 8: Programmatic Usage Patterns")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    asc_file = project_root / "old/sim_files/PT1000_circuit_1.asc"

    if not asc_file.exists():
        print(f"File not found: {asc_file}")
        return

    result = parse_asc_file(asc_file)

    # Pattern 1: Find components with parametric values
    print("\nComponents with parametric values:")
    for comp in result.components:
        value = comp.value
        if value and value.startswith("{"):
            print(f"  {comp.ref}: {value}")

    # Pattern 2: Build a component lookup dictionary
    print("\nComponent lookup dictionary:")
    comp_dict = {c.ref: c for c in result.components if c.ref}
    print(f"  Created dict with {len(comp_dict)} components")

    # Pattern 3: Extract measurement names
    print("\nMeasurement names:")
    meas_names = [m.name for m in result.measurements]
    print(f"  {meas_names}")

    # Pattern 4: Get the transient analysis parameters
    print("\nTransient analysis:")
    for cmd in result.analysis_commands:
        if cmd.analysis_type == "tran":
            print(f"  .tran {cmd.parameters}")

    # Pattern 5: Summary dictionary
    print("\nSummary dictionary:")
    summary = result.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")


def main():
    """Run all demos."""
    demo_basic_parsing()
    demo_component_analysis()
    demo_parameter_extraction()
    demo_net_analysis()
    demo_wire_analysis()
    demo_compare_schematics()
    demo_raw_data_access()
    demo_programmatic_usage()

    print("\n" + "=" * 70)
    print("All demos completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
