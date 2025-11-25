"""Wien Bridge Oscillator

A classic sine wave oscillator using positive and negative feedback.

Run: python examples/analog/wien_bridge_osc.py
"""

import math

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def build_wien_bridge_osc(freq: float = 1000) -> Circuit:
    """Build a Wien bridge oscillator.

    Args:
        freq: Oscillation frequency in Hz

    Returns:
        Circuit with the oscillator
    """
    circuit = Circuit("wien_bridge_osc")

    # Power supplies
    vcc = Vdc("cc", 15.0)
    vee = Vdc("ee", -15.0)

    # Calculate R and C for desired frequency
    # f = 1 / (2π × R × C)
    C = 10e-9  # Choose C = 10nF
    R = 1 / (2 * math.pi * freq * C)

    # Wien bridge network (frequency determining)
    r1 = Resistor("1", resistance=R)  # Series R
    c1 = Capacitor("1", capacitance=C)  # Series C
    r2 = Resistor("2", resistance=R)  # Shunt R
    c2 = Capacitor("2", capacitance=C)  # Shunt C

    # Gain setting network (for gain = 3)
    # Need gain slightly > 3 for oscillation to start
    rf = Resistor("f", resistance=22_000)
    rg = Resistor("g", resistance=10_000)

    # Amplitude limiting diodes (soft limiting)
    d1 = create_component("diode.1n4148", "1")
    d2 = create_component("diode.1n4148", "2")
    r_limit = Resistor("limit", resistance=47_000)

    opamp = create_component("opamp.tl072", "1")

    circuit.add(vcc, vee, r1, c1, r2, c2, rf, rg, d1, d2, r_limit, opamp)

    # Nets
    vcc_net = Net("vcc")
    vee_net = Net("vee")
    vout = Net("vout")
    vplus = Net("vplus")
    vminus = Net("vminus")
    v_rc = Net("v_rc")

    # Power supplies
    circuit.connect(vcc.ports[0], vcc_net)
    circuit.connect(vcc.ports[1], GND)
    circuit.connect(vee.ports[0], GND)
    circuit.connect(vee.ports[1], vee_net)

    # Wien bridge positive feedback network
    # Output -> R1 -> C1 -> vplus -> shunt R2||C2 -> GND
    circuit.connect(r1.ports[0], vout)
    circuit.connect(r1.ports[1], v_rc)
    circuit.connect(c1.ports[0], v_rc)
    circuit.connect(c1.ports[1], vplus)
    circuit.connect(r2.ports[0], vplus)
    circuit.connect(r2.ports[1], GND)
    circuit.connect(c2.ports[0], vplus)
    circuit.connect(c2.ports[1], GND)

    # Negative feedback network with amplitude limiting
    circuit.connect(rf.ports[0], vout)
    circuit.connect(rf.ports[1], vminus)
    circuit.connect(rg.ports[0], vminus)
    circuit.connect(rg.ports[1], GND)

    # Diode limiting (antiparallel diodes in parallel with Rf)
    circuit.connect(r_limit.ports[0], vout)
    circuit.connect(r_limit.ports[1], vminus)
    circuit.connect(d1.ports[0], vminus)  # D1: vminus to vout
    circuit.connect(d1.ports[1], vout)
    circuit.connect(d2.ports[0], vout)  # D2: vout to vminus
    circuit.connect(d2.ports[1], vminus)

    # Op-amp
    circuit.connect(opamp.ports[0], vout)
    circuit.connect(opamp.ports[1], vplus)
    circuit.connect(opamp.ports[2], vminus)
    circuit.connect(opamp.ports[3], vcc_net)
    circuit.connect(opamp.ports[4], vee_net)

    return circuit, R, C


def main():
    """Demonstrate Wien bridge oscillator."""
    print("=" * 60)
    print("Wien Bridge Oscillator")
    print("=" * 60)

    freq = 1000
    circuit, R, C = build_wien_bridge_osc(freq=freq)

    print(f"""
   Wien Bridge Oscillator

   Circuit topology:

                     ┌───[R1]──[C1]───┐
                     │                │
   Vout ────────────┼────────────(+)─┤
         │          │                │
         │          ├──[R2]──GND     │
         │          │                │
         │          └──[C2]──GND    [OP]── Vout
         │                           │
         └──[Rf]─┬─(-)───────────────┘
                 │
                [Rg]
                 │
                GND

   Design:
   - Oscillation frequency: f = 1/(2π×R×C)
   - f = 1/(2π × {R:.0f} × {C*1e9:.1f}nF) = {freq} Hz

   Component values:
   - R1 = R2 = {R:.0f} Ω
   - C1 = C2 = {C*1e9:.1f} nF
   - Rf = 22kΩ, Rg = 10kΩ (gain ≈ 3.2)

   Oscillation condition:
   - Loop gain = 1 at oscillation frequency
   - Wien network gain = 1/3 at f0
   - Amplifier gain must be ≥ 3

   Amplitude stabilization:
   - Diodes provide soft limiting
   - As amplitude grows, diode conduction reduces effective Rf
   - Settles at ~2Vpp typical
""")

    print("   Netlist:")
    print(circuit.build_netlist())

    result = circuit.validate()
    print(f"\n   Validation: {'VALID' if result.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
