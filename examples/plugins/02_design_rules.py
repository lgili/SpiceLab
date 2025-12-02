"""Example: Design Rules Check (DRC) Plugin

This example demonstrates how to use the DesignRulesPlugin to
validate circuits before simulation and catch common design errors.

The plugin can detect:
- Missing ground connections
- Floating nodes
- Invalid component values
- Shorted voltage sources
- Power limit violations
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Inductor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.plugins.examples import DesignRulesPlugin


def create_valid_circuit() -> Circuit:
    """Create a properly designed circuit."""
    circuit = Circuit("valid_rc_filter")

    vin = Vdc("Vin", 5.0)
    r1 = Resistor("R1", "1k")
    c1 = Capacitor("C1", "1u")

    circuit.add(vin, r1, c1)

    circuit.connect(vin.ports[0], Net("in"))
    circuit.connect(vin.ports[1], GND)
    circuit.connect(r1.ports[0], Net("in"))
    circuit.connect(r1.ports[1], Net("out"))
    circuit.connect(c1.ports[0], Net("out"))
    circuit.connect(c1.ports[1], GND)

    return circuit


def create_circuit_missing_ground() -> Circuit:
    """Create a circuit with no ground reference."""
    circuit = Circuit("no_ground")

    r1 = Resistor("R1", "1k")
    r2 = Resistor("R2", "2k")

    circuit.add(r1, r2)

    # No connection to GND!
    circuit.connect(r1.ports[0], Net("a"))
    circuit.connect(r1.ports[1], Net("b"))
    circuit.connect(r2.ports[0], Net("b"))
    circuit.connect(r2.ports[1], Net("c"))

    return circuit


def create_circuit_floating_node() -> Circuit:
    """Create a circuit with a floating node."""
    circuit = Circuit("floating_node")

    vin = Vdc("Vin", 5.0)
    r1 = Resistor("R1", "1k")
    r2 = Resistor("R2", "2k")  # This one will have a floating terminal

    circuit.add(vin, r1, r2)

    circuit.connect(vin.ports[0], Net("in"))
    circuit.connect(vin.ports[1], GND)
    circuit.connect(r1.ports[0], Net("in"))
    circuit.connect(r1.ports[1], GND)
    # R2 only connected on one side - floating!
    circuit.connect(r2.ports[0], Net("floating"))
    circuit.connect(r2.ports[1], GND)

    return circuit


def create_circuit_low_resistance() -> Circuit:
    """Create a circuit with suspiciously low resistance."""
    circuit = Circuit("low_resistance")

    vin = Vdc("Vin", 5.0)
    r1 = Resistor("R1", "0.0001")  # 0.1 mOhm - very low!

    circuit.add(vin, r1)

    circuit.connect(vin.ports[0], Net("in"))
    circuit.connect(vin.ports[1], GND)
    circuit.connect(r1.ports[0], Net("in"))
    circuit.connect(r1.ports[1], GND)

    return circuit


def example_basic_drc():
    """Run basic DRC on valid and invalid circuits."""
    print("=" * 60)
    print("Example 1: Basic Design Rules Check")
    print("=" * 60)

    plugin = DesignRulesPlugin()

    # Check valid circuit
    print("\n1. Checking valid circuit:")
    valid_circuit = create_valid_circuit()
    result = plugin.check(valid_circuit)
    print(f"   Result: {result}")
    if result.passed:
        print("   ✓ Circuit passes all checks!")

    # Check circuit missing ground
    print("\n2. Checking circuit with missing ground:")
    no_ground = create_circuit_missing_ground()
    result = plugin.check(no_ground)
    print(f"   Result: {result}")
    for v in result.violations:
        print(f"   - {v}")

    # Check circuit with floating node
    print("\n3. Checking circuit with floating node:")
    floating = create_circuit_floating_node()
    result = plugin.check(floating)
    print(f"   Result: {result}")
    for v in result.violations:
        print(f"   - {v}")


def example_configure_rules():
    """Configure which rules to enable/disable."""
    print("\n" + "=" * 60)
    print("Example 2: Configure DRC Rules")
    print("=" * 60)

    plugin = DesignRulesPlugin()

    # Default configuration
    print("\nDefault rules enabled:")
    print("  - floating_nodes: True")
    print("  - short_circuits: True")
    print("  - missing_ground: True")
    print("  - component_values: True")

    # Disable specific rules
    print("\nConfiguring to disable floating_nodes check:")
    plugin.configure({
        "rules": {
            "floating_nodes": False,
        }
    })

    floating = create_circuit_floating_node()
    result = plugin.check(floating)
    print(f"Result (floating_nodes disabled): {result}")
    print(f"Violations: {len(result.violations)}")

    # Re-enable and check again
    plugin.configure({
        "rules": {
            "floating_nodes": True,
        }
    })

    result = plugin.check(floating)
    print(f"\nResult (floating_nodes enabled): {result}")


def example_component_value_checks():
    """Check component value validation."""
    print("\n" + "=" * 60)
    print("Example 3: Component Value Validation")
    print("=" * 60)

    plugin = DesignRulesPlugin()

    # Configure min/max values
    plugin.configure({
        "rules": {
            "min_resistance": 1e-3,  # 1 mOhm minimum
            "max_capacitance": 0.1,  # 100 mF maximum
        }
    })

    print("\nChecking circuit with very low resistance:")
    low_r = create_circuit_low_resistance()
    result = plugin.check(low_r)
    for v in result.violations:
        if "resistance" in v.rule_name.lower():
            print(f"   {v}")
            if v.suggestion:
                print(f"   Suggestion: {v.suggestion}")


def example_severity_levels():
    """Demonstrate different severity levels."""
    print("\n" + "=" * 60)
    print("Example 4: Understanding Severity Levels")
    print("=" * 60)

    plugin = DesignRulesPlugin()

    no_ground = create_circuit_missing_ground()
    result = plugin.check(no_ground)

    print("\nViolations by severity:")
    from spicelab.plugins.examples.design_rules_plugin import RuleSeverity

    errors = [v for v in result.violations if v.severity == RuleSeverity.ERROR]
    warnings = [v for v in result.violations if v.severity == RuleSeverity.WARNING]
    infos = [v for v in result.violations if v.severity == RuleSeverity.INFO]

    print(f"\n  ERRORS ({len(errors)}):")
    for v in errors:
        print(f"    - {v.rule_name}: {v.message}")

    print(f"\n  WARNINGS ({len(warnings)}):")
    for v in warnings:
        print(f"    - {v.rule_name}: {v.message}")

    print(f"\n  INFO ({len(infos)}):")
    for v in infos:
        print(f"    - {v.rule_name}: {v.message}")


def example_block_simulation_on_error():
    """Configure DRC to block simulation on errors."""
    print("\n" + "=" * 60)
    print("Example 5: Block Simulation on DRC Errors")
    print("=" * 60)

    plugin = DesignRulesPlugin()

    # Configure to block simulation on errors
    plugin.configure({
        "block_on_error": True,
    })

    print("\nWith block_on_error=True, if DRC fails during pre-simulation hook,")
    print("it will raise a RuntimeError and prevent the simulation from running.")
    print("\nThis is useful for CI/CD pipelines to catch design errors early.")

    # Activate the plugin
    plugin.activate()

    # The plugin is now registered to run before each simulation
    print("\nPlugin activated - DRC will run before each simulation.")


def example_drc_workflow():
    """Show a complete DRC workflow."""
    print("\n" + "=" * 60)
    print("Example 6: Complete DRC Workflow")
    print("=" * 60)

    plugin = DesignRulesPlugin()

    # Create a circuit
    circuit = Circuit("my_design")
    vin = Vdc("Vin", 12.0)
    r1 = Resistor("R1", "4.7k")
    r2 = Resistor("R2", "10k")
    c1 = Capacitor("C1", "100n")
    ind = Inductor("L1", "10m")

    circuit.add(vin, r1, r2, c1, ind)

    circuit.connect(vin.ports[0], Net("vcc"))
    circuit.connect(vin.ports[1], GND)
    circuit.connect(r1.ports[0], Net("vcc"))
    circuit.connect(r1.ports[1], Net("bias"))
    circuit.connect(r2.ports[0], Net("bias"))
    circuit.connect(r2.ports[1], GND)
    circuit.connect(c1.ports[0], Net("bias"))
    circuit.connect(c1.ports[1], Net("filtered"))
    circuit.connect(ind.ports[0], Net("filtered"))
    circuit.connect(ind.ports[1], GND)

    print("\nCircuit: my_design")
    print(f"Components: {len(circuit._components)}")

    # Run DRC
    print("\nRunning DRC...")
    result = plugin.check(circuit)

    print(f"\nDRC Result: {'PASS' if result.passed else 'FAIL'}")
    print(f"  Errors: {result.errors}")
    print(f"  Warnings: {result.warnings}")

    if result.violations:
        print("\nIssues found:")
        for v in result.violations:
            print(f"  [{v.severity.value.upper()}] {v.rule_name}")
            print(f"    {v.message}")
            if v.suggestion:
                print(f"    → {v.suggestion}")

    # Save last result for reference
    last_result = plugin.get_last_result()
    print(f"\nLast DRC result stored: {last_result is not None}")


if __name__ == "__main__":
    example_basic_drc()
    example_configure_rules()
    example_component_value_checks()
    example_severity_levels()
    example_block_simulation_on_error()
    example_drc_workflow()

    print("\n" + "=" * 60)
    print("Design Rules Plugin Examples Complete!")
    print("=" * 60)
