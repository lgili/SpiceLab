"""Tutorial 01: Your First Circuit

This tutorial teaches you how to:
1. Create a simple circuit (voltage divider)
2. Add components and connect them
3. Run a DC simulation
4. View the results

Run: python examples/tutorials/01_first_circuit.py
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND, Net


def main():
    """Build and simulate a simple voltage divider."""
    print("=" * 60)
    print("Tutorial 01: Your First Circuit - Voltage Divider")
    print("=" * 60)

    # =========================================================================
    # Step 1: Create the circuit
    # =========================================================================
    print("\n1. Creating circuit...")
    circuit = Circuit("voltage_divider")

    # =========================================================================
    # Step 2: Create components
    # =========================================================================
    print("2. Creating components...")

    # Voltage source: 10V DC
    v1 = Vdc("1", 10.0)

    # Resistors for voltage divider (R1 = R2 = 10k => Vout = 5V)
    r1 = Resistor("1", resistance=10_000)  # 10k ohm
    r2 = Resistor("2", resistance=10_000)  # 10k ohm

    # Add components to circuit
    circuit.add(v1, r1, r2)

    # =========================================================================
    # Step 3: Create nets and connect components
    # =========================================================================
    print("3. Connecting components...")

    # Create named nets
    vin = Net("vin")  # Input voltage node
    vout = Net("vout")  # Output voltage node (middle of divider)

    # Connect voltage source: V1+ to vin, V1- to GND
    circuit.connect(v1.ports[0], vin)
    circuit.connect(v1.ports[1], GND)

    # Connect R1: between vin and vout
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], vout)

    # Connect R2: between vout and GND
    circuit.connect(r2.ports[0], vout)
    circuit.connect(r2.ports[1], GND)

    # =========================================================================
    # Step 4: View the netlist
    # =========================================================================
    print("4. Generated SPICE netlist:")
    print("-" * 40)
    print(circuit.build_netlist())
    print("-" * 40)

    # =========================================================================
    # Step 5: Validate the circuit
    # =========================================================================
    print("\n5. Validating circuit...")
    result = circuit.validate()
    print(f"   Validation: {'PASSED' if result.is_valid else 'FAILED'}")
    if result.errors:
        for err in result.errors:
            print(f"   Error: {err.message}")
    if result.warnings:
        for warn in result.warnings:
            print(f"   Warning: {warn.message}")

    # =========================================================================
    # Step 6: Calculate expected output
    # =========================================================================
    print("\n6. Expected results:")
    vin_value = 10.0
    r1_value = 10_000
    r2_value = 10_000
    vout_expected = vin_value * r2_value / (r1_value + r2_value)
    print(f"   Vin = {vin_value}V")
    print(f"   R1 = {r1_value/1000}k, R2 = {r2_value/1000}k")
    print(f"   Vout = Vin * R2/(R1+R2) = {vout_expected}V")

    print("\n" + "=" * 60)
    print("Congratulations! You've built your first circuit.")
    print("=" * 60)

    # =========================================================================
    # Key concepts learned:
    # =========================================================================
    print("""
Key Concepts:
- Circuit: Container for components and connections
- Vdc: DC voltage source with positive and negative terminals
- Resistor: Two-terminal resistor component
- Net: Named connection point (like a wire)
- GND: Special ground reference net (node 0 in SPICE)
- ports[0], ports[1]: Component terminal access
- connect(): Links a component port to a net
- build_netlist(): Generates SPICE-format netlist
- validate(): Checks for common circuit errors
""")


if __name__ == "__main__":
    main()
