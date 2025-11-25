"""Current Mirror

A BJT current mirror - fundamental current source building block.

Run: python examples/analog/current_mirror.py
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def build_simple_current_mirror(i_ref: float = 1e-3) -> Circuit:
    """Build a simple two-transistor current mirror.

    Args:
        i_ref: Reference current in Amperes

    Returns:
        Circuit with the current mirror
    """
    circuit = Circuit("current_mirror")

    # Power supply
    vcc = Vdc("cc", 12.0)

    # Reference resistor to set Iref
    # Rref = (Vcc - Vbe) / Iref
    vbe = 0.7
    R_ref = (12.0 - vbe) / i_ref

    r_ref = Resistor("ref", resistance=R_ref)

    # Load resistor (to demonstrate current copying)
    r_load = Resistor("load", resistance=1000)

    # Matched transistor pair
    q1 = create_component("bjt.2n3904", "1")  # Reference (diode-connected)
    q2 = create_component("bjt.2n3904", "2")  # Output mirror

    circuit.add(vcc, r_ref, r_load, q1, q2)

    # Nets
    vcc_net = Net("vcc")
    v_mirror = Net("v_mirror")
    v_out = Net("v_out")

    # Power supply
    circuit.connect(vcc.ports[0], vcc_net)
    circuit.connect(vcc.ports[1], GND)

    # Reference side: Vcc -> Rref -> Q1 (diode connected)
    circuit.connect(r_ref.ports[0], vcc_net)
    circuit.connect(r_ref.ports[1], v_mirror)

    # Q1: diode-connected (collector tied to base)
    circuit.connect(q1.ports[0], v_mirror)  # Collector
    circuit.connect(q1.ports[1], v_mirror)  # Base (tied to collector)
    circuit.connect(q1.ports[2], GND)  # Emitter

    # Q2: mirror output
    circuit.connect(q2.ports[1], v_mirror)  # Base (same as Q1)
    circuit.connect(q2.ports[0], v_out)  # Collector (output)
    circuit.connect(q2.ports[2], GND)  # Emitter

    # Load on output
    circuit.connect(r_load.ports[0], vcc_net)
    circuit.connect(r_load.ports[1], v_out)

    return circuit, R_ref


def build_wilson_current_mirror(i_ref: float = 1e-3) -> Circuit:
    """Build a Wilson current mirror (improved accuracy).

    Args:
        i_ref: Reference current in Amperes

    Returns:
        Circuit with the Wilson mirror
    """
    circuit = Circuit("wilson_mirror")

    # Power supply
    vcc = Vdc("cc", 12.0)

    # Reference resistor
    vbe = 0.7
    R_ref = (12.0 - 2 * vbe) / i_ref  # Two Vbe drops

    r_ref = Resistor("ref", resistance=R_ref)
    r_load = Resistor("load", resistance=1000)

    # Three transistors for Wilson
    q1 = create_component("bjt.2n3904", "1")
    q2 = create_component("bjt.2n3904", "2")
    q3 = create_component("bjt.2n3904", "3")

    circuit.add(vcc, r_ref, r_load, q1, q2, q3)

    # Nets
    vcc_net = Net("vcc")
    v_base = Net("v_base")
    v_q3e = Net("v_q3e")
    v_out = Net("v_out")

    # Power supply
    circuit.connect(vcc.ports[0], vcc_net)
    circuit.connect(vcc.ports[1], GND)

    # Reference resistor
    circuit.connect(r_ref.ports[0], vcc_net)
    circuit.connect(r_ref.ports[1], v_q3e)

    # Q3: input transistor (collector provides Iref)
    circuit.connect(q3.ports[0], v_q3e)  # Collector
    circuit.connect(q3.ports[1], v_base)  # Base
    circuit.connect(q3.ports[2], v_base)  # Emitter tied to base node

    # Q1: diode-connected
    circuit.connect(q1.ports[0], v_base)  # Collector
    circuit.connect(q1.ports[1], v_base)  # Base
    circuit.connect(q1.ports[2], GND)  # Emitter

    # Q2: output mirror
    circuit.connect(q2.ports[0], v_out)  # Collector (output)
    circuit.connect(q2.ports[1], v_base)  # Base
    circuit.connect(q2.ports[2], GND)  # Emitter

    # Load
    circuit.connect(r_load.ports[0], vcc_net)
    circuit.connect(r_load.ports[1], v_out)

    return circuit, R_ref


def main():
    """Demonstrate current mirror circuits."""
    print("=" * 60)
    print("Current Mirror Circuits")
    print("=" * 60)

    i_ref = 1e-3  # 1mA

    print("\n1. Simple Current Mirror")
    print("-" * 40)

    circuit1, R_ref1 = build_simple_current_mirror(i_ref)

    print(f"""
   Simple two-transistor current mirror

   Circuit:
        Vcc
         │
       [Rref]
         │
         ├──────────┐
         │          │
        ─┤Q1      Q2├─
     C──B│          │B──C
         E          E
         │          │
        GND        GND

   Design:
   - Rref = (Vcc - Vbe) / Iref = {R_ref1:.0f} Ω
   - Iref = 1mA

   Operation:
   - Q1 is diode-connected (Vbe = Vbe)
   - Q2 base sees same Vbe
   - With matched transistors: Iout ≈ Iref

   Accuracy: ~5-10% due to:
   - Early effect (Vce different for Q1, Q2)
   - Base current error
""")
    print("   Netlist:")
    print(circuit1.build_netlist())

    print("\n2. Wilson Current Mirror")
    print("-" * 40)

    circuit2, R_ref2 = build_wilson_current_mirror(i_ref)

    print(f"""
   Wilson current mirror (improved)

   Circuit:
        Vcc
         │
       [Rref]
         │
         ├──Q3──┐
         │  │   │
         │  E   C
         │      │
         ├──────┤
         │      │
        ─┤Q1  Q2├─
         │      │
        GND    GND

   Rref = {R_ref2:.0f} Ω (accounts for 2 Vbe drops)

   Advantages:
   - Higher output impedance
   - Better current matching (~1%)
   - Reduced base current error
   - Q3 provides feedback to improve accuracy
""")
    print("   Netlist:")
    print(circuit2.build_netlist())


if __name__ == "__main__":
    main()
