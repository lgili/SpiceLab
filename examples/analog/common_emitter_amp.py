"""Common Emitter Amplifier

A classic BJT amplifier configuration with voltage gain.

Run: python examples/analog/common_emitter_amp.py
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def build_common_emitter_amp() -> Circuit:
    """Build a common emitter amplifier with bias network."""
    circuit = Circuit("common_emitter_amp")

    # Power supply
    vcc = Vdc("cc", 12.0)

    # Bias network (voltage divider)
    r1 = Resistor("1", resistance=47_000)  # Top bias resistor
    r2 = Resistor("2", resistance=10_000)  # Bottom bias resistor
    rc = Resistor("c", resistance=2_200)  # Collector resistor
    re = Resistor("e", resistance=470)  # Emitter resistor

    # AC coupling capacitors
    cin = Capacitor("in", capacitance=10e-6)  # Input coupling
    cout = Capacitor("out", capacitance=10e-6)  # Output coupling
    ce = Capacitor("e", capacitance=100e-6)  # Emitter bypass

    # BJT from library
    q1 = create_component("bjt.2n3904", "1")

    circuit.add(vcc, r1, r2, rc, re, cin, cout, ce, q1)

    # Nets
    vcc_net = Net("vcc")
    vin = Net("vin")
    vbase = Net("vbase")
    vcollector = Net("vcollector")
    vemitter = Net("vemitter")
    vout = Net("vout")

    # Power supply
    circuit.connect(vcc.ports[0], vcc_net)
    circuit.connect(vcc.ports[1], GND)

    # Bias network
    circuit.connect(r1.ports[0], vcc_net)
    circuit.connect(r1.ports[1], vbase)
    circuit.connect(r2.ports[0], vbase)
    circuit.connect(r2.ports[1], GND)

    # Collector circuit
    circuit.connect(rc.ports[0], vcc_net)
    circuit.connect(rc.ports[1], vcollector)

    # BJT connections (C, B, E)
    circuit.connect(q1.ports[0], vcollector)
    circuit.connect(q1.ports[1], vbase)
    circuit.connect(q1.ports[2], vemitter)

    # Emitter resistor and bypass
    circuit.connect(re.ports[0], vemitter)
    circuit.connect(re.ports[1], GND)
    circuit.connect(ce.ports[0], vemitter)
    circuit.connect(ce.ports[1], GND)

    # Input/output coupling
    circuit.connect(cin.ports[0], vin)
    circuit.connect(cin.ports[1], vbase)
    circuit.connect(cout.ports[0], vcollector)
    circuit.connect(cout.ports[1], vout)

    return circuit


def main():
    """Demonstrate common emitter amplifier."""
    print("=" * 60)
    print("Common Emitter Amplifier")
    print("=" * 60)

    circuit = build_common_emitter_amp()

    print("""
   Circuit topology:

        Vcc (12V)
         │
        [R1] 47k
         │
         ├──[Cin]── Vin
         │
         ├─── Vbase
         │
        [R2] 10k        [Rc] 2.2k
         │               │
        GND              ├──[Cout]── Vout
                         │
                    ──┤ Q1 (2N3904)
                         │
                        [Re] 470Ω ──┬── GND
                                    │
                                   [Ce] 100µF

   Design equations:
   - Vbase = Vcc × R2/(R1+R2) = 12 × 10k/57k ≈ 2.1V
   - Ve = Vbase - 0.7V ≈ 1.4V
   - Ie ≈ Ve/Re = 1.4V/470Ω ≈ 3mA
   - Vc = Vcc - Ic×Rc ≈ 12 - 3m×2.2k ≈ 5.4V

   Voltage gain (with Ce):
   - Av ≈ -Rc/re' where re' = 26mV/Ie ≈ 8.7Ω
   - Av ≈ -2200/8.7 ≈ -253 (inverted)

   Without Ce (Re not bypassed):
   - Av ≈ -Rc/Re = -2200/470 ≈ -4.7
""")

    print("   Netlist:")
    print(circuit.build_netlist())

    result = circuit.validate()
    print(f"\n   Validation: {'VALID' if result.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
