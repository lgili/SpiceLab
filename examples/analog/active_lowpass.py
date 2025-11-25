"""Active Low-Pass Filter

A Sallen-Key 2nd order low-pass filter using an op-amp.

Run: python examples/analog/active_lowpass.py
"""

import math

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def build_sallen_key_lowpass(fc: float = 1000, q: float = 0.707) -> Circuit:
    """Build a Sallen-Key lowpass filter.

    Args:
        fc: Cutoff frequency in Hz
        q: Quality factor (0.707 for Butterworth response)

    Returns:
        Circuit with the filter
    """
    circuit = Circuit("sallen_key_lp")

    # Power supplies
    vcc = Vdc("cc", 15.0)
    vee = Vdc("ee", -15.0)

    # Calculate component values for unity gain Sallen-Key
    # Using equal R, equal C design
    # fc = 1/(2π×R×C), Q = 0.5 (for unity gain, equal components)
    # For Q = 0.707, we need to adjust
    R = 10_000  # Choose R
    C = 1 / (2 * math.pi * R * fc)

    # Components
    r1 = Resistor("1", resistance=R)
    r2 = Resistor("2", resistance=R)
    c1 = Capacitor("1", capacitance=C)
    c2 = Capacitor("2", capacitance=C)

    opamp = create_component("opamp.tl072", "1")

    circuit.add(vcc, vee, r1, r2, c1, c2, opamp)

    # Nets
    vcc_net = Net("vcc")
    vee_net = Net("vee")
    vin = Net("vin")
    v1 = Net("v1")
    v2 = Net("v2")
    vout = Net("vout")

    # Power supplies
    circuit.connect(vcc.ports[0], vcc_net)
    circuit.connect(vcc.ports[1], GND)
    circuit.connect(vee.ports[0], GND)
    circuit.connect(vee.ports[1], vee_net)

    # Input R network
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], v1)
    circuit.connect(r2.ports[0], v1)
    circuit.connect(r2.ports[1], v2)

    # Capacitors
    circuit.connect(c1.ports[0], v1)
    circuit.connect(c1.ports[1], vout)
    circuit.connect(c2.ports[0], v2)
    circuit.connect(c2.ports[1], GND)

    # Op-amp (out, +in, -in, V+, V-)
    circuit.connect(opamp.ports[0], vout)
    circuit.connect(opamp.ports[1], v2)  # + input
    circuit.connect(opamp.ports[2], vout)  # - input (unity gain)
    circuit.connect(opamp.ports[3], vcc_net)
    circuit.connect(opamp.ports[4], vee_net)

    return circuit, R, C


def main():
    """Demonstrate active lowpass filter."""
    print("=" * 60)
    print("Sallen-Key Active Low-Pass Filter")
    print("=" * 60)

    fc = 1000
    circuit, R, C = build_sallen_key_lowpass(fc=fc)

    print(f"""
   Sallen-Key 2nd Order Lowpass Filter

   Circuit topology:

   Vin ──[R1]──┬──[R2]──┬────(+)───┐
               │        │          │
              [C1]     [C2]       [OP]── Vout
               │        │          │     │
               └────────│──────────┘     │
                        │                │
                       GND      (-)──────┘

   Design (unity gain, equal components):
   - fc = {fc} Hz
   - R1 = R2 = {R} Ω
   - C1 = C2 = {C*1e9:.2f} nF

   Transfer function:
   H(s) = ωc² / (s² + (ωc/Q)×s + ωc²)

   Characteristics:
   - 2nd order: -40 dB/decade roll-off
   - Unity gain in passband
   - Q = 0.5 with equal components (slightly underdamped)
   - Butterworth (Q=0.707) needs component adjustment

   Advantages:
   - Only one op-amp needed
   - Non-inverting
   - Easy to cascade for higher orders
""")

    print("   Netlist:")
    print(circuit.build_netlist())

    result = circuit.validate()
    print(f"\n   Validation: {'VALID' if result.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
