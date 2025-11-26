#!/usr/bin/env python3
"""Demonstration of loading an ASC file and running a simulation.

This example shows how to:
1. Load and parse an LTspice .asc file
2. Run simulation directly using LTspice (default engine for ASC files)
3. Access all measurements defined in the file
4. Display simulation results

Run: python examples/asc_simulation_demo.py

Requires LTspice installed (default) or ngspice for alternative engine.
"""

from __future__ import annotations

from pathlib import Path

from spicelab.io import (
    get_analyses_from_asc,
    load_circuit_from_asc,
    parse_asc_file,
    print_asc_summary,
    print_conversion_result,
    run_asc_simulation,
)


def demo_simple_usage():
    """Demonstrate the simplest way to run an ASC simulation."""
    print("=" * 70)
    print("DEMO 1: Simple ASC Simulation (one line!)")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    asc_file = project_root / "old/sim_files/PT1000_circuit_1.asc"

    if not asc_file.exists():
        print(f"File not found: {asc_file}")
        return

    print(f"\nRunning simulation for: {asc_file.name}")
    print("Using LTspice (default engine for ASC files)")
    print("-" * 50)

    try:
        # One line to load ASC and run simulation!
        result = run_asc_simulation(str(asc_file))

        ds = result.dataset()
        print("\nSimulation completed!")
        print(f"Variables: {list(ds.data_vars.keys())}")

    except Exception as e:
        print(f"\nSimulation failed: {e}")
        print("\nNote: Make sure LTspice is installed.")
        print("Or use engine='ngspice' if you have ngspice installed.")


def demo_parse_and_inspect():
    """Demonstrate parsing and inspecting an ASC file."""
    print("\n" + "=" * 70)
    print("DEMO 2: Parse and Inspect ASC File")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    asc_file = project_root / "old/sim_files/PT1000_circuit_1.asc"

    if not asc_file.exists():
        print(f"File not found: {asc_file}")
        return

    # Parse ASC file
    print(f"\nParsing: {asc_file.name}")
    print("-" * 50)
    asc_result = parse_asc_file(asc_file)
    print_asc_summary(asc_result)

    # Show analyses (already parsed by the library!)
    print("\nAnalysis commands (auto-parsed):")
    print("-" * 50)
    analyses = get_analyses_from_asc(asc_result)
    for analysis in analyses:
        print(f"  Mode: {analysis['mode']}")
        print(f"  Args: {analysis['args']}")
        print()


def demo_convert_and_netlist():
    """Demonstrate converting ASC to Circuit and generating netlist."""
    print("\n" + "=" * 70)
    print("DEMO 3: Convert to Circuit and Generate Netlist")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    asc_file = project_root / "old/sim_files/PT1000_circuit_1.asc"

    if not asc_file.exists():
        print(f"File not found: {asc_file}")
        return

    # Convert to SpiceLab Circuit
    print(f"\nConverting: {asc_file.name}")
    print("-" * 50)
    conv_result = load_circuit_from_asc(str(asc_file))
    print_conversion_result(conv_result)

    # Generate netlist
    print("Generated Netlist:")
    print("-" * 50)
    netlist = conv_result.circuit.build_netlist()
    print(netlist)


def demo_measurements():
    """Demonstrate accessing measurements from ASC file."""
    print("\n" + "=" * 70)
    print("DEMO 4: Measurements Defined in ASC File")
    print("=" * 70)

    project_root = Path(__file__).parent.parent
    asc_file = project_root / "old/sim_files/PT1000_circuit_1.asc"

    if not asc_file.exists():
        print(f"File not found: {asc_file}")
        return

    asc_result = parse_asc_file(asc_file)

    print(f"\nFile: {asc_file.name}")

    # Show parameters
    print("\n" + "-" * 50)
    print("SPICE PARAMETERS (.param)")
    print("-" * 50)
    for name, param in asc_result.parameters.items():
        print(f"  {name:15} = {param.value}")

    # Show all measurements
    print("\n" + "-" * 50)
    print("MEASUREMENTS DEFINED (.meas)")
    print("-" * 50)
    print(f"  Total: {len(asc_result.measurements)} measurements")
    print()
    for i, meas in enumerate(asc_result.measurements, 1):
        print(f"  {i:2}. {meas.name:15} = .meas {meas.measurement_type} {meas.expression}")

    # Categorize by type
    meas_types: dict[str, list[str]] = {}
    for meas in asc_result.measurements:
        mtype = meas.measurement_type.upper()
        if mtype not in meas_types:
            meas_types[mtype] = []
        meas_types[mtype].append(meas.name)

    print("\n  By type:")
    for mtype, names in sorted(meas_types.items()):
        print(f"    {mtype}: {', '.join(names)}")


def demo_ngspice_fallback():
    """Demonstrate using ngspice as alternative engine."""
    print("\n" + "=" * 70)
    print("DEMO 5: Using ngspice as Alternative Engine")
    print("=" * 70)

    # Build a simple circuit manually (to avoid LTspice-specific syntax)
    from spicelab.core.circuit import Circuit
    from spicelab.core.components import Capacitor, Resistor, Vpulse
    from spicelab.core.net import GND, Net
    from spicelab.core.types import AnalysisSpec
    from spicelab.engines.factory import create_simulator

    circuit = Circuit("simple_rc")

    # Create components
    v1 = Vpulse("1", v1=0, v2=5, td=0, tr=1e-9, tf=1e-9, pw=1e-3, per=2e-3)
    r1 = Resistor("1", "10k")
    c1 = Capacitor("1", "100n")

    circuit.add(v1, r1, c1)

    # Connect
    vin = Net("vin")
    vout = Net("vout")
    circuit.connect(v1.ports[0], vin)
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], vout)
    circuit.connect(c1.ports[0], vout)
    circuit.connect(c1.ports[1], GND)

    print("\nCircuit: Simple RC filter")
    print("  V1: Vpulse 0V to 5V")
    print("  R1: 10k")
    print("  C1: 100nF")
    print(f"  Time constant: {10e3 * 100e-9 * 1000:.1f} ms")

    print("\nNetlist:")
    print("-" * 40)
    print(circuit.build_netlist())

    # Run with ngspice
    spec = AnalysisSpec("tran", {"tstep": 1e-6, "tstop": 5e-3})

    print(f"Running {spec.mode} analysis with ngspice...")

    try:
        sim = create_simulator("ngspice")
        handle = sim.run(circuit, [spec], None, None)
        ds = handle.dataset()

        print("\nSimulation completed!")
        print(f"Time points: {len(ds.coords.get('time', []))}")
        print(f"Variables: {list(ds.data_vars.keys())}")

        if "V(vout)" in ds.data_vars:
            vout_data = ds["V(vout)"].values
            print("\nResults:")
            print(f"  V(vout) max: {vout_data.max():.4f}V")
            print(f"  V(vout) min: {vout_data.min():.4f}V")
            print(f"  V(vout) avg: {vout_data.mean():.4f}V")

    except Exception as e:
        print(f"\nSimulation failed: {e}")


def main():
    """Run all demos."""
    demo_simple_usage()
    demo_parse_and_inspect()
    demo_convert_and_netlist()
    demo_measurements()
    demo_ngspice_fallback()

    print("\n" + "=" * 70)
    print("All demos completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
