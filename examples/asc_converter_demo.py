#!/usr/bin/env python3
"""Demonstration of ASC to Circuit conversion.

This example shows how to:
1. Parse an LTspice .asc file
2. Convert it to SpiceLab Circuit objects
3. Handle warnings for unsupported components
4. Generate SPICE netlists from the converted circuit

The conversion process maps LTspice schematic symbols to SpiceLab components
and attempts to preserve the circuit topology through geometric wire analysis.
"""

from pathlib import Path

from spicelab.io import (
    asc_to_circuit,
    load_circuit_from_asc,
    parse_asc_file,
    print_asc_summary,
    print_conversion_result,
)


def demo_basic_conversion():
    """Demonstrate basic ASC file conversion."""
    print("=" * 70)
    print("DEMO 1: Basic ASC to Circuit Conversion")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    asc_file = project_root / "old/sim_files/PT1000_circuit_1.asc"

    if not asc_file.exists():
        print(f"File not found: {asc_file}")
        return

    # Method 1: Two-step conversion (parse then convert)
    print(f"\nParsing: {asc_file.name}")
    print("-" * 50)

    asc_result = parse_asc_file(asc_file)
    print_asc_summary(asc_result)

    print("\nConverting to SpiceLab Circuit...")
    print("-" * 50)

    conv_result = asc_to_circuit(asc_result)
    print_conversion_result(conv_result)


def demo_load_and_convert():
    """Demonstrate one-step load and convert."""
    print("\n" + "=" * 70)
    print("DEMO 2: One-Step Load and Convert")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    asc_file = project_root / "old/sim_files/PT1000_circuit_1.asc"

    if not asc_file.exists():
        print(f"File not found: {asc_file}")
        return

    # Method 2: One-step conversion
    result = load_circuit_from_asc(str(asc_file))

    print(f"\nConversion summary for {asc_file.name}:")
    summary = result.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")


def demo_warnings_and_skipped():
    """Demonstrate warning handling for unsupported components."""
    print("\n" + "=" * 70)
    print("DEMO 3: Warning Handling for Unsupported Components")
    print("=" * 70)

    project_root = Path(__file__).parent.parent

    # Load all ASC files and check for warnings
    asc_files = list(project_root.glob("old/sim_files/*.asc"))

    if not asc_files:
        print("No .asc files found")
        return

    for asc_file in asc_files:
        result = load_circuit_from_asc(str(asc_file))

        print(f"\n{asc_file.name}:")
        print(f"  Converted: {len(result.converted_components)}")
        print(f"  Skipped: {len(result.skipped_components)}")
        print(f"  Warnings: {len(result.warnings)}")

        if result.warnings:
            print("  Warning details:")
            for w in result.warnings:
                print(f"    - {w}")


def demo_netlist_generation():
    """Demonstrate netlist generation from converted circuit."""
    print("\n" + "=" * 70)
    print("DEMO 4: Netlist Generation")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    asc_file = project_root / "old/sim_files/PT1000_circuit_1.asc"

    if not asc_file.exists():
        print(f"File not found: {asc_file}")
        return

    result = load_circuit_from_asc(str(asc_file))

    print(f"\nGenerated netlist for {asc_file.name}:")
    print("-" * 50)

    netlist = result.circuit.build_netlist()
    print(netlist)


def demo_component_access():
    """Demonstrate accessing converted components."""
    print("\n" + "=" * 70)
    print("DEMO 5: Accessing Converted Components")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    asc_file = project_root / "old/sim_files/PT1000_circuit_1.asc"

    if not asc_file.exists():
        print(f"File not found: {asc_file}")
        return

    result = load_circuit_from_asc(str(asc_file))

    print(f"\nConverted components from {asc_file.name}:")
    print("-" * 50)

    # Access components by reference
    for ref, comp in result.component_map.items():
        print(f"  {ref}: {type(comp).__name__}")
        print(f"    Value: {comp.value}")
        print(f"    Ports: {len(comp.ports)}")
        print(f"    Ports names: {[p.name for p in comp.ports]}")
        print()


def demo_compare_original_vs_converted():
    """Compare original ASC data with converted circuit."""
    print("\n" + "=" * 70)
    print("DEMO 6: Original ASC vs Converted Circuit")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    asc_file = project_root / "old/sim_files/PT1000_circuit_1.asc"

    if not asc_file.exists():
        print(f"File not found: {asc_file}")
        return

    # Parse original
    asc_result = parse_asc_file(asc_file)

    # Convert
    conv_result = asc_to_circuit(asc_result)

    print(f"\nComparison for {asc_file.name}:")
    print("-" * 50)

    print("\nOriginal ASC data:")
    print(f"  Total components: {len(asc_result.components)}")
    print(f"  Parameters: {len(asc_result.parameters)}")
    print(f"  Measurements: {len(asc_result.measurements)}")
    print(f"  Analysis commands: {len(asc_result.analysis_commands)}")

    print("\nConverted Circuit:")
    print(f"  Components converted: {len(conv_result.converted_components)}")
    print(f"  Components skipped: {len(conv_result.skipped_components)}")
    conversion_rate = len(conv_result.converted_components) / len(asc_result.components) * 100
    print(f"  Conversion rate: {conversion_rate:.1f}%")

    # Show what was skipped
    if conv_result.skipped_components:
        print("\n  Skipped components:")
        for ref in conv_result.skipped_components:
            # Find original component
            for asc_comp in asc_result.components:
                if asc_comp.ref == ref:
                    print(f"    {ref}: {asc_comp.symbol}")
                    break


def demo_simple_rc_circuit():
    """Demonstrate conversion with a simple inline ASC."""
    print("\n" + "=" * 70)
    print("DEMO 7: Simple RC Circuit (inline ASC)")
    print("=" * 70)

    # Create a simple RC circuit ASC string
    asc_content = """Version 4
SHEET 1 880 680
WIRE 100 100 100 200
WIRE 100 200 200 200
WIRE 200 200 200 300
FLAG 100 100 Vin
FLAG 200 300 0
SYMBOL voltage 50 100 R0
SYMATTR InstName V1
SYMATTR Value 5
SYMBOL res 100 100 R0
SYMATTR InstName R1
SYMATTR Value 10k
SYMBOL cap 200 200 R0
SYMATTR InstName C1
SYMATTR Value 100n
TEXT 50 400 Left 2 !.tran 0 1m 0
"""

    from spicelab.io import asc_to_circuit, parse_asc_string

    # Parse and convert
    asc_result = parse_asc_string(asc_content)
    conv_result = asc_to_circuit(asc_result, circuit_name="simple_rc")

    print("\nParsed ASC content:")
    print(f"  Components: {len(asc_result.components)}")
    for comp in asc_result.components:
        print(f"    {comp.ref}: {comp.symbol} = {comp.value}")

    print("\nConversion result:")
    print(f"  Converted: {len(conv_result.converted_components)}")
    print(f"  Warnings: {len(conv_result.warnings)}")

    print("\nGenerated netlist:")
    print("-" * 40)
    print(conv_result.circuit.build_netlist())


def demo_supported_symbols():
    """Show all supported symbols for conversion."""
    print("\n" + "=" * 70)
    print("DEMO 8: Supported Symbols Reference")
    print("=" * 70)

    from spicelab.io import KNOWN_UNSUPPORTED, SYMBOL_MAP

    print("\nSupported symbols (can be converted):")
    print("-" * 50)
    for symbol, comp_class in sorted(SYMBOL_MAP.items()):
        if comp_class is not None:
            print(f"  {symbol:20} -> {comp_class.__name__}")

    print("\n\nKnown unsupported symbols (will generate warnings):")
    print("-" * 50)
    for symbol, reason in sorted(KNOWN_UNSUPPORTED.items()):
        print(f"  {symbol:15} - {reason}")


def main():
    """Run all demos."""
    demo_basic_conversion()
    demo_load_and_convert()
    demo_warnings_and_skipped()
    demo_netlist_generation()
    demo_component_access()
    demo_compare_original_vs_converted()
    demo_simple_rc_circuit()
    demo_supported_symbols()

    print("\n" + "=" * 70)
    print("All demos completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
