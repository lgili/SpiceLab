"""DX Demo: Interactive Troubleshooting

Demonstrates how to diagnose and fix common simulation problems.

Troubleshooter features:
- Auto-diagnose circuit issues
- Convergence failure analysis
- Empty results diagnosis
- Guided fix suggestions

Run: python examples/dx_troubleshooting_demo.py
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.troubleshooting import (
    Troubleshooter,
    diagnose_convergence,
    diagnose_empty_results,
)


def demo_auto_diagnose():
    """Demonstrate automatic circuit diagnostics."""
    print("=" * 60)
    print("1. Auto-Diagnose Circuit Issues")
    print("=" * 60)

    # Create a circuit with potential problems
    c = Circuit("problematic")
    v = Vdc("1", 5.0)
    r_small = Resistor("1", resistance=0.0001)  # Very small
    r_large = Resistor("2", resistance=1e15)  # Very large
    cap_small = Capacitor("1", capacitance=1e-18)  # Very small
    c.add(v, r_small, r_large, cap_small)

    n1 = Net("n1")
    n2 = Net("n2")
    c.connect(v.ports[0], n1)
    c.connect(r_small.ports[0], n1)
    c.connect(r_small.ports[1], n2)
    c.connect(r_large.ports[0], n2)
    c.connect(r_large.ports[1], GND)
    c.connect(cap_small.ports[0], n2)
    c.connect(cap_small.ports[1], GND)
    c.connect(v.ports[1], GND)

    print("\nCircuit has extreme component values that may cause issues.")
    print("\nRunning diagnostics...")

    ts = Troubleshooter(c)
    ts.diagnose()


def demo_convergence_diagnosis():
    """Demonstrate convergence failure diagnosis."""
    print("\n" + "=" * 60)
    print("2. Convergence Failure Diagnosis")
    print("=" * 60)

    c = Circuit("convergence_test")
    v = Vdc("1", 5.0)
    r = Resistor("1", resistance=1000)
    c.add(v, r)

    c.connect(v.ports[0], Net("vcc"))
    c.connect(r.ports[0], Net("vcc"))
    c.connect(v.ports[1], GND)
    c.connect(r.ports[1], GND)

    print("\nSimulating a convergence error...")
    print("Error message: 'timestep too small in transient analysis'")

    # Diagnose with error message
    result = diagnose_convergence(c, "timestep too small in transient analysis")

    print(f"\nDiagnosis found {len(result.findings)} findings:")
    for finding in result.findings:
        print(f"\n  [{finding.severity.value.upper()}] {finding.message}")
        print(f"  Suggestion: {finding.suggestion}")
        if finding.details:
            print(f"  Details: {finding.details}")


def demo_empty_results_diagnosis():
    """Demonstrate empty results diagnosis."""
    print("\n" + "=" * 60)
    print("3. Empty Results Diagnosis")
    print("=" * 60)

    c = Circuit("empty_results_test")
    v = Vdc("1", 5.0)
    r = Resistor("1", resistance=1000)
    c.add(v, r)

    c.connect(v.ports[0], Net("vcc"))
    c.connect(r.ports[0], Net("vcc"))
    c.connect(v.ports[1], GND)
    c.connect(r.ports[1], GND)

    print("\nDiagnosing why simulation returned empty results...")

    # Case 1: No probes
    print("\n--- Case 1: No probes specified ---")
    result = diagnose_empty_results(c, analysis_type="tran", probes=[])
    for finding in result.findings:
        print(f"  [{finding.severity.value.upper()}] {finding.message}")

    # Case 2: DC analysis
    print("\n--- Case 2: DC analysis (single point) ---")
    result = diagnose_empty_results(c, analysis_type="dc", probes=["V(vcc)"])
    for finding in result.findings:
        print(f"  [{finding.severity.value.upper()}] {finding.message}")


def demo_quick_fix():
    """Demonstrate quick fix suggestions."""
    print("\n" + "=" * 60)
    print("4. Quick Fix Suggestions")
    print("=" * 60)

    # Create a circuit with issues
    c = Circuit("needs_fixes")
    r1 = Resistor("1", resistance=1000)
    r2 = Resistor("2", resistance=2000)
    c.add(r1, r2)

    # Floating nodes
    c.connect(r1.ports[0], Net("a"))
    c.connect(r1.ports[1], Net("b"))
    c.connect(r2.ports[0], Net("c"))
    c.connect(r2.ports[1], Net("d"))

    print("\nCircuit has floating nodes and no ground.")
    print("\nQuick fix suggestions:")

    ts = Troubleshooter(c)
    fixes = ts.quick_fix()

    for i, fix in enumerate(fixes, 1):
        print(f"\n  {i}. {fix}")


def demo_troubleshooter_cli():
    """Show CLI usage examples."""
    print("\n" + "=" * 60)
    print("5. CLI Usage")
    print("=" * 60)

    print("""
The troubleshooter is also available as a CLI command:

  # Auto-diagnose circuit
  spicelab-troubleshoot circuit.py

  # Focus on convergence issues
  spicelab-troubleshoot circuit.py --convergence

  # With error message from simulator
  spicelab-troubleshoot circuit.py -c -e "timestep too small"

  # Interactive troubleshooting session
  spicelab-troubleshoot circuit.py --interactive

  # Just get quick fix suggestions
  spicelab-troubleshoot circuit.py --quick-fix

  # JSON output for scripting
  spicelab-troubleshoot circuit.py --json
""")


def main():
    """Run all troubleshooting demos."""
    print("\n" + "=" * 60)
    print("SpiceLab Troubleshooting Demo")
    print("=" * 60)
    print("\nThe troubleshooter helps diagnose and fix common simulation")
    print("problems with helpful suggestions and guided fixes.")

    demo_auto_diagnose()
    demo_convergence_diagnosis()
    demo_empty_results_diagnosis()
    demo_quick_fix()
    demo_troubleshooter_cli()

    print("\n" + "=" * 60)
    print("Use the troubleshooter when simulations fail!")
    print("=" * 60)


if __name__ == "__main__":
    main()
