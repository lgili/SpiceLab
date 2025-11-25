"""Differential Amplifier

A BJT differential pair amplifier - fundamental building block.

Run: python examples/analog/differential_amp.py
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def build_differential_amp() -> Circuit:
    """Build a differential amplifier with current source tail."""
    circuit = Circuit("differential_amp")

    # Power supplies
    vcc = Vdc("cc", 15.0)
    vee = Vdc("ee", -15.0)

    # Input voltages (for DC analysis)
    vin_p = Vdc("in_p", 0.0)  # Positive input
    vin_n = Vdc("in_n", 0.0)  # Negative input

    # Collector resistors (matched)
    rc1 = Resistor("c1", resistance=10_000)
    rc2 = Resistor("c2", resistance=10_000)

    # Tail current source resistor
    re = Resistor("e", resistance=15_000)  # Sets ~1mA tail current

    # Matched transistor pair
    q1 = create_component("bjt.2n3904", "1")
    q2 = create_component("bjt.2n3904", "2")

    circuit.add(vcc, vee, vin_p, vin_n, rc1, rc2, re, q1, q2)

    # Nets
    vcc_net = Net("vcc")
    vee_net = Net("vee")
    vinp_net = Net("vinp")
    vinn_net = Net("vinn")
    vout_p = Net("vout_p")
    vout_n = Net("vout_n")
    vtail = Net("vtail")

    # Power supplies
    circuit.connect(vcc.ports[0], vcc_net)
    circuit.connect(vcc.ports[1], GND)
    circuit.connect(vee.ports[0], GND)
    circuit.connect(vee.ports[1], vee_net)

    # Input sources
    circuit.connect(vin_p.ports[0], vinp_net)
    circuit.connect(vin_p.ports[1], GND)
    circuit.connect(vin_n.ports[0], vinn_net)
    circuit.connect(vin_n.ports[1], GND)

    # Collector resistors
    circuit.connect(rc1.ports[0], vcc_net)
    circuit.connect(rc1.ports[1], vout_p)
    circuit.connect(rc2.ports[0], vcc_net)
    circuit.connect(rc2.ports[1], vout_n)

    # Q1 (C, B, E)
    circuit.connect(q1.ports[0], vout_p)
    circuit.connect(q1.ports[1], vinp_net)
    circuit.connect(q1.ports[2], vtail)

    # Q2 (C, B, E)
    circuit.connect(q2.ports[0], vout_n)
    circuit.connect(q2.ports[1], vinn_net)
    circuit.connect(q2.ports[2], vtail)

    # Tail resistor
    circuit.connect(re.ports[0], vtail)
    circuit.connect(re.ports[1], vee_net)

    return circuit


def main():
    """Demonstrate differential amplifier."""
    print("=" * 60)
    print("Differential Amplifier")
    print("=" * 60)

    circuit = build_differential_amp()

    print("""
   Circuit topology:

        Vcc (+15V)
         │       │
        [Rc1]   [Rc2]
         │       │
    Vout+├       ├Vout-
         │       │
    Vin+─┤Q1   Q2├─Vin-
              │
            vtail
              │
            [Re]
              │
        Vee (-15V)

   Operation:
   - Differential input: Vdiff = Vin+ - Vin-
   - Common mode input: Vcm = (Vin+ + Vin-) / 2
   - Amplifies differential, rejects common mode

   Design:
   - Tail current: Itail = (|Vee| - Vbe) / Re
   - Itail ≈ (15V - 0.7V) / 15kΩ ≈ 0.95mA
   - Each transistor: Ic ≈ Itail/2 ≈ 0.48mA

   Gain:
   - Differential gain: Ad = gm × Rc = (Ic/Vt) × Rc
   - Ad ≈ (0.48mA/26mV) × 10kΩ ≈ 185

   Common Mode Rejection:
   - CMRR ≈ gm × Re (with resistor tail)
   - Much higher with current source tail
""")

    print("   Netlist:")
    print(circuit.build_netlist())

    result = circuit.validate()
    print(f"\n   Validation: {'VALID' if result.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
