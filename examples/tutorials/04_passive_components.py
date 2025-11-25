"""Tutorial 04: Passive Components

This tutorial teaches you how to use passive components:
1. Resistors with different values
2. Capacitors and their behavior
3. Inductors and their behavior
4. Series and parallel combinations

Run: python examples/tutorials/04_passive_components.py
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Inductor, Resistor, Vdc
from spicelab.core.net import GND, Net


def demo_resistors():
    """Demonstrate resistor usage."""
    print("\n1. Resistors")
    print("-" * 40)

    circuit = Circuit("resistor_demo")

    # Various resistor values
    r1 = Resistor("1", resistance=1000)  # 1k ohm
    r2 = Resistor("2", resistance=10_000)  # 10k ohm
    r3 = Resistor("3", resistance=100_000)  # 100k ohm

    circuit.add(r1, r2, r3)

    # Connect in series
    v1 = Net("v1")
    v2 = Net("v2")
    v3 = Net("v3")

    circuit.connect(r1.ports[0], v1)
    circuit.connect(r1.ports[1], v2)
    circuit.connect(r2.ports[0], v2)
    circuit.connect(r2.ports[1], v3)
    circuit.connect(r3.ports[0], v3)
    circuit.connect(r3.ports[1], GND)

    print("""
   Resistor: Two-terminal passive component

   Usage: Resistor(ref, resistance=1000)

   Parameters:
   - resistance: Value in Ohms

   Common values (E12 series):
   1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2

   Series combination: R_total = R1 + R2 + R3
   Parallel combination: 1/R_total = 1/R1 + 1/R2 + 1/R3
""")
    print("   Netlist (3 resistors in series):")
    print(circuit.build_netlist())


def demo_capacitors():
    """Demonstrate capacitor usage."""
    print("\n2. Capacitors")
    print("-" * 40)

    circuit = Circuit("capacitor_demo")

    # Various capacitor values
    c1 = Capacitor("1", capacitance=100e-12)  # 100pF
    c2 = Capacitor("2", capacitance=100e-9)  # 100nF
    c3 = Capacitor("3", capacitance=100e-6)  # 100uF

    circuit.add(c1, c2, c3)

    # Connect in parallel (common in bypass applications)
    vcc = Net("vcc")
    circuit.connect(c1.ports[0], vcc)
    circuit.connect(c1.ports[1], GND)
    circuit.connect(c2.ports[0], vcc)
    circuit.connect(c2.ports[1], GND)
    circuit.connect(c3.ports[0], vcc)
    circuit.connect(c3.ports[1], GND)

    print("""
   Capacitor: Stores energy in electric field

   Usage: Capacitor(ref, capacitance=100e-9)

   Parameters:
   - capacitance: Value in Farads

   Common values:
   - pF range: High-frequency bypass, RF circuits
   - nF range: Signal coupling, filtering
   - uF range: Power supply bypass, timing

   Impedance: Xc = 1 / (2π × f × C)

   Series combination: 1/C_total = 1/C1 + 1/C2 + 1/C3
   Parallel combination: C_total = C1 + C2 + C3
""")
    print("   Netlist (3 capacitors in parallel for bypass):")
    print(circuit.build_netlist())


def demo_inductors():
    """Demonstrate inductor usage."""
    print("\n3. Inductors")
    print("-" * 40)

    circuit = Circuit("inductor_demo")

    # Various inductor values
    l1 = Inductor("1", inductance=1e-6)  # 1uH
    l2 = Inductor("2", inductance=100e-6)  # 100uH
    l3 = Inductor("3", inductance=10e-3)  # 10mH

    circuit.add(l1, l2, l3)

    # Connect in series (like a choke chain)
    vin = Net("vin")
    v1 = Net("v1")
    v2 = Net("v2")

    circuit.connect(l1.ports[0], vin)
    circuit.connect(l1.ports[1], v1)
    circuit.connect(l2.ports[0], v1)
    circuit.connect(l2.ports[1], v2)
    circuit.connect(l3.ports[0], v2)
    circuit.connect(l3.ports[1], GND)

    print("""
   Inductor: Stores energy in magnetic field

   Usage: Inductor(ref, inductance=100e-6)

   Parameters:
   - inductance: Value in Henries

   Common values:
   - uH range: RF filters, switching supplies
   - mH range: Audio filters, power supplies
   - H range: Power line filters

   Impedance: XL = 2π × f × L

   Series combination: L_total = L1 + L2 + L3
   Parallel combination: 1/L_total = 1/L1 + 1/L2 + 1/L3
""")
    print("   Netlist (3 inductors in series):")
    print(circuit.build_netlist())


def demo_rlc_circuit():
    """Demonstrate RLC circuit."""
    print("\n4. RLC Series Circuit")
    print("-" * 40)

    circuit = Circuit("rlc_series")

    v1 = Vdc("1", 10.0)
    r1 = Resistor("1", resistance=100)
    l1 = Inductor("1", inductance=10e-3)  # 10mH
    c1 = Capacitor("1", capacitance=100e-9)  # 100nF

    circuit.add(v1, r1, l1, c1)

    # Create nets
    vin = Net("vin")
    v_rl = Net("v_rl")
    v_lc = Net("v_lc")

    # Connect in series: V -> R -> L -> C -> GND
    circuit.connect(v1.ports[0], vin)
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], v_rl)
    circuit.connect(l1.ports[0], v_rl)
    circuit.connect(l1.ports[1], v_lc)
    circuit.connect(c1.ports[0], v_lc)
    circuit.connect(c1.ports[1], GND)

    # Calculate resonant frequency
    import math

    L = 10e-3
    C = 100e-9
    f0 = 1 / (2 * math.pi * math.sqrt(L * C))

    print(f"""
   RLC Series: Resonant circuit

   Resonant frequency: f0 = 1 / (2π × √(L × C))
   f0 = 1 / (2π × √({L} × {C}))
   f0 = {f0:.0f} Hz

   At resonance:
   - XL = XC (inductive and capacitive reactances cancel)
   - Impedance is minimum (equals R only)
   - Current is maximum
   - Phase angle is 0°

   Quality factor: Q = (1/R) × √(L/C)
""")
    print("   Netlist:")
    print(circuit.build_netlist())


def main():
    """Demonstrate passive components."""
    print("=" * 60)
    print("Tutorial 04: Passive Components")
    print("=" * 60)

    demo_resistors()
    demo_capacitors()
    demo_inductors()
    demo_rlc_circuit()

    print("\n" + "=" * 60)
    print("Summary: Component Prefixes in SPICE")
    print("=" * 60)
    print("""
   | Prefix | Component  | Unit    | Example    |
   |--------|------------|---------|------------|
   | R      | Resistor   | Ohm     | R1 1k      |
   | C      | Capacitor  | Farad   | C1 100n    |
   | L      | Inductor   | Henry   | L1 10m     |
   | V      | Voltage    | Volt    | V1 DC 5    |
   | I      | Current    | Ampere  | I1 DC 1m   |
""")


if __name__ == "__main__":
    main()
