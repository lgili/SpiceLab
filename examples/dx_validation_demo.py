"""DX Demo: Circuit Validation

Demonstrates how to validate circuits before simulation to catch common errors.

Validation checks:
- Ground reference (required for DC operating point)
- Floating nodes (nodes with only one connection)
- Voltage source loops (parallel voltage sources)
- Component value warnings (extreme values)

Run: python examples/dx_validation_demo.py
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net


def demo_valid_circuit():
    """Demonstrate validation of a valid circuit."""
    print("=" * 60)
    print("1. Valid Circuit")
    print("=" * 60)

    c = Circuit("valid_rc_filter")
    v = Vdc("1", 5.0)
    r = Resistor("1", resistance=10_000)
    cap = Capacitor("1", capacitance=100e-9)
    c.add(v, r, cap)

    # Proper connections
    vin = Net("vin")
    vout = Net("vout")
    c.connect(v.ports[0], vin)
    c.connect(r.ports[0], vin)
    c.connect(r.ports[1], vout)
    c.connect(cap.ports[0], vout)
    c.connect(cap.ports[1], GND)
    c.connect(v.ports[1], GND)

    result = c.validate()
    print(f"\nCircuit: {c.name}")
    print(f"Components: {len(c._components)}")
    print("\nValidation Result:")
    print(result)


def demo_floating_nodes():
    """Demonstrate detection of floating nodes."""
    print("\n" + "=" * 60)
    print("2. Floating Nodes Detection")
    print("=" * 60)

    c = Circuit("floating_nodes")
    r1 = Resistor("1", resistance=1000)
    r2 = Resistor("2", resistance=2000)
    c.add(r1, r2)

    # Each resistor on separate nets - all floating!
    c.connect(r1.ports[0], Net("a"))
    c.connect(r1.ports[1], Net("b"))
    c.connect(r2.ports[0], Net("c"))
    c.connect(r2.ports[1], Net("d"))

    result = c.validate()
    print(f"\nCircuit: {c.name}")
    print("Problem: Each net has only one connection")
    print("\nValidation Result:")
    print(result)


def demo_no_ground():
    """Demonstrate detection of missing ground."""
    print("\n" + "=" * 60)
    print("3. Missing Ground Detection")
    print("=" * 60)

    c = Circuit("no_ground")
    v = Vdc("1", 5.0)
    r = Resistor("1", resistance=1000)
    c.add(v, r)

    # No connection to GND
    n1 = Net("n1")
    n2 = Net("n2")
    c.connect(v.ports[0], n1)
    c.connect(r.ports[0], n1)
    c.connect(v.ports[1], n2)
    c.connect(r.ports[1], n2)

    result = c.validate()
    print(f"\nCircuit: {c.name}")
    print("Problem: No component connected to GND")
    print("\nValidation Result:")
    print(result)


def demo_voltage_source_loop():
    """Demonstrate detection of parallel voltage sources."""
    print("\n" + "=" * 60)
    print("4. Voltage Source Loop Detection")
    print("=" * 60)

    c = Circuit("parallel_v_sources")
    v1 = Vdc("1", 5.0)
    v2 = Vdc("2", 10.0)  # Different voltage!
    r = Resistor("1", resistance=1000)
    c.add(v1, v2, r)

    # Both sources in parallel - this is an error!
    vcc = Net("vcc")
    c.connect(v1.ports[0], vcc)
    c.connect(v2.ports[0], vcc)  # Both V+ on same node
    c.connect(v1.ports[1], GND)
    c.connect(v2.ports[1], GND)  # Both V- on GND
    c.connect(r.ports[0], vcc)
    c.connect(r.ports[1], GND)

    result = c.validate()
    print(f"\nCircuit: {c.name}")
    print("Problem: V1 (5V) and V2 (10V) are in parallel")
    print("\nValidation Result:")
    print(result)


def demo_extreme_values():
    """Demonstrate warnings for extreme component values."""
    print("\n" + "=" * 60)
    print("5. Extreme Values Warning")
    print("=" * 60)

    c = Circuit("extreme_values")
    v = Vdc("1", 5.0)
    r_tiny = Resistor("1", resistance=0.0001)  # 0.1mΩ - very small!
    r_huge = Resistor("2", resistance=1e15)  # 1PΩ - very large!
    c.add(v, r_tiny, r_huge)

    n1 = Net("n1")
    n2 = Net("n2")
    c.connect(v.ports[0], n1)
    c.connect(r_tiny.ports[0], n1)
    c.connect(r_tiny.ports[1], n2)
    c.connect(r_huge.ports[0], n2)
    c.connect(r_huge.ports[1], GND)
    c.connect(v.ports[1], GND)

    result = c.validate()
    print(f"\nCircuit: {c.name}")
    print("Problem: R1=0.1mΩ (too small), R2=1PΩ (too large)")
    print("\nValidation Result:")
    print(result)


def demo_strict_mode():
    """Demonstrate strict validation mode."""
    print("\n" + "=" * 60)
    print("6. Strict Mode")
    print("=" * 60)

    c = Circuit("strict_test")
    v = Vdc("1", 5.0)
    r = Resistor("1", resistance=0.0001)  # Warning: very small
    c.add(v, r)

    n1 = Net("n1")
    c.connect(v.ports[0], n1)
    c.connect(r.ports[0], n1)
    c.connect(v.ports[1], GND)
    c.connect(r.ports[1], GND)

    normal = c.validate(strict=False)
    strict = c.validate(strict=True)

    print("\nNormal mode (warnings allowed):")
    print(f"  is_valid: {normal.is_valid}")

    print("\nStrict mode (warnings become errors):")
    print(f"  is_valid: {strict.is_valid}")


def main():
    """Run all validation demos."""
    print("\n" + "=" * 60)
    print("SpiceLab Circuit Validation Demo")
    print("=" * 60)
    print("\nValidation catches common circuit errors BEFORE simulation,")
    print("saving time and providing helpful suggestions.")

    demo_valid_circuit()
    demo_floating_nodes()
    demo_no_ground()
    demo_voltage_source_loop()
    demo_extreme_values()
    demo_strict_mode()

    print("\n" + "=" * 60)
    print("Always validate circuits before simulation!")
    print("CLI: spicelab-validate circuit.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
