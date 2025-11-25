"""Multiple Feedback Bandpass Filter

An active bandpass filter using the multiple feedback topology.

Run: python examples/analog/bandpass_filter.py
"""

import math

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def build_mfb_bandpass(f0: float = 1000, q: float = 10, gain: float = 1) -> Circuit:
    """Build a multiple feedback bandpass filter.

    Args:
        f0: Center frequency in Hz
        q: Quality factor (bandwidth = f0/Q)
        gain: Passband gain at center frequency

    Returns:
        Circuit with the filter
    """
    circuit = Circuit("mfb_bandpass")

    # Power supplies
    vcc = Vdc("cc", 15.0)
    vee = Vdc("ee", -15.0)

    # Calculate component values
    # Using equal capacitor design
    C = 10e-9  # Choose C = 10nF

    # MFB bandpass equations:
    # f0 = 1/(2π×C) × √(1/(R1×R3))
    # Q = π × f0 × C × R2
    # Gain = -R2/(2×R1)

    # Solving:
    R2 = q / (math.pi * f0 * C)
    R1 = R2 / (2 * abs(gain))
    R3 = R1 / (4 * q * q * abs(gain) - 1) if (4 * q * q * abs(gain)) > 1 else R1

    # Components
    r1 = Resistor("1", resistance=R1)
    r2 = Resistor("2", resistance=R2)
    r3 = Resistor("3", resistance=R3)
    c1 = Capacitor("1", capacitance=C)
    c2 = Capacitor("2", capacitance=C)

    opamp = create_component("opamp.tl072", "1")

    circuit.add(vcc, vee, r1, r2, r3, c1, c2, opamp)

    # Nets
    vcc_net = Net("vcc")
    vee_net = Net("vee")
    vin = Net("vin")
    vminus = Net("vminus")
    vout = Net("vout")

    # Power supplies
    circuit.connect(vcc.ports[0], vcc_net)
    circuit.connect(vcc.ports[1], GND)
    circuit.connect(vee.ports[0], GND)
    circuit.connect(vee.ports[1], vee_net)

    # Input through R1
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], vminus)

    # Feedback from output through R2 and C1 (parallel)
    circuit.connect(r2.ports[0], vout)
    circuit.connect(r2.ports[1], vminus)
    circuit.connect(c1.ports[0], vout)
    circuit.connect(c1.ports[1], vminus)

    # R3 and C2 to ground
    circuit.connect(r3.ports[0], vminus)
    circuit.connect(r3.ports[1], GND)
    circuit.connect(c2.ports[0], vminus)
    circuit.connect(c2.ports[1], GND)

    # Op-amp (out, +in, -in, V+, V-)
    circuit.connect(opamp.ports[0], vout)
    circuit.connect(opamp.ports[1], GND)  # + input grounded
    circuit.connect(opamp.ports[2], vminus)  # - input (inverting)
    circuit.connect(opamp.ports[3], vcc_net)
    circuit.connect(opamp.ports[4], vee_net)

    return circuit, R1, R2, R3, C


def main():
    """Demonstrate bandpass filter."""
    print("=" * 60)
    print("Multiple Feedback Bandpass Filter")
    print("=" * 60)

    f0 = 1000
    Q = 10
    circuit, R1, R2, R3, C = build_mfb_bandpass(f0=f0, q=Q)
    bw = f0 / Q

    print(f"""
   Multiple Feedback (MFB) Bandpass Filter

   Circuit topology:

               R2
            ┌──┤├──┐
            │  C1  │
   Vin ─[R1]─┼─||──┼─(-)──┐
             │     │      │
            [R3]  [C2]   [OP]── Vout
             │     │      │
            GND   GND   (+)
                         │
                        GND

   Design parameters:
   - Center frequency: f0 = {f0} Hz
   - Quality factor: Q = {Q}
   - Bandwidth: BW = f0/Q = {bw} Hz
   - Passband: {f0-bw/2:.0f} Hz to {f0+bw/2:.0f} Hz

   Component values:
   - R1 = {R1:.0f} Ω
   - R2 = {R2:.0f} Ω
   - R3 = {R3:.0f} Ω
   - C1 = C2 = {C*1e9:.1f} nF

   Characteristics:
   - Inverted output (180° phase at f0)
   - -20 dB/decade on both sides
   - Higher Q = narrower bandwidth
   - Gain at f0 = -R2/(2×R1)
""")

    print("   Netlist:")
    print(circuit.build_netlist())

    result = circuit.validate()
    print(f"\n   Validation: {'VALID' if result.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
