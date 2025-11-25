"""Allpass Filter (Phase Shifter)

An allpass filter passes all frequencies with unity gain but shifts
the phase. Used for phase equalization, audio effects, and creating
notch filters by combining with the original signal.

Run: python examples/signal_processing/allpass_filter.py
"""

import math

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vac
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def build_first_order_allpass(f0: float = 1000.0) -> Circuit:
    """Build a first-order allpass filter.

    At f0, the phase shift is -90 degrees.
    Below f0: phase approaches 0 degrees
    Above f0: phase approaches -180 degrees

    Args:
        f0: Frequency where phase = -90 degrees

    Returns:
        Circuit with the allpass filter
    """
    circuit = Circuit("allpass_1st_order")

    # f0 = 1 / (2 * pi * R * C)
    C = 10e-9  # 10nF
    R = 1 / (2 * math.pi * f0 * C)

    # Input source
    v_in = Vac("in", ac_mag=1.0)

    # Op-amp
    u1 = create_component("opamp.ideal", "1")

    # Resistors
    r1 = Resistor("1", resistance=R)  # To capacitor
    r2 = Resistor("2", resistance=R)  # Feedback
    r3 = Resistor("3", resistance=R)  # Input divider
    r4 = Resistor("4", resistance=R)  # Input divider

    # Capacitor
    c1 = Capacitor("1", capacitance=C)

    circuit.add(v_in, u1, r1, r2, r3, r4, c1)

    # Nets
    vin = Net("vin")
    vout = Net("vout")
    v_inv = Net("v_inv")
    v_noninv = Net("v_noninv")

    # Input
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)

    # Input divider to non-inverting input
    circuit.connect(r3.ports[0], vin)
    circuit.connect(r3.ports[1], v_noninv)
    circuit.connect(r4.ports[0], v_noninv)
    circuit.connect(r4.ports[1], GND)

    # RC to non-inverting (creates frequency-dependent divider)
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], v_noninv)
    circuit.connect(c1.ports[0], v_noninv)
    circuit.connect(c1.ports[1], GND)

    # Op-amp connections
    circuit.connect(u1.ports[0], v_inv)  # Inverting input
    circuit.connect(u1.ports[1], v_noninv)  # Non-inverting input
    circuit.connect(u1.ports[2], vout)  # Output

    # Feedback network
    circuit.connect(r2.ports[0], vout)
    circuit.connect(r2.ports[1], v_inv)

    # Input to inverting through resistor (for unity gain)
    r5 = Resistor("5", resistance=R)
    circuit.add(r5)
    circuit.connect(r5.ports[0], vin)
    circuit.connect(r5.ports[1], v_inv)

    return circuit, R, C


def build_second_order_allpass(f0: float = 1000.0, Q: float = 0.707) -> Circuit:
    """Build a second-order allpass filter.

    At f0, the phase shift is -180 degrees.
    Provides steeper phase transition than first-order.

    Args:
        f0: Center frequency
        Q: Quality factor (affects transition steepness)

    Returns:
        Circuit with the second-order allpass filter
    """
    circuit = Circuit("allpass_2nd_order")

    # Use twin-T based topology for 2nd order
    C = 10e-9
    R = 1 / (2 * math.pi * f0 * C)

    # Input source
    v_in = Vac("in", ac_mag=1.0)

    # Op-amp
    u1 = create_component("opamp.ideal", "1")

    # Twin-T resistors
    r1 = Resistor("1", resistance=R)
    r2 = Resistor("2", resistance=R)
    r3 = Resistor("3", resistance=R / 2)  # Center resistor

    # Twin-T capacitors
    c1 = Capacitor("1", capacitance=C)
    c2 = Capacitor("2", capacitance=C)
    c3 = Capacitor("3", capacitance=2 * C)  # Center capacitor

    # Feedback resistors for Q control
    r_fb1 = Resistor("fb1", resistance=10_000)
    r_fb2 = Resistor("fb2", resistance=10_000 / (2 * Q))

    circuit.add(v_in, u1, r1, r2, r3, c1, c2, c3, r_fb1, r_fb2)

    # Nets
    vin = Net("vin")
    vout = Net("vout")
    v_inv = Net("v_inv")
    v_mid_r = Net("mid_r")
    v_mid_c = Net("mid_c")

    # Input
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)

    # R-C-R path of twin-T
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], v_mid_r)
    circuit.connect(r2.ports[0], v_mid_r)
    circuit.connect(r2.ports[1], vout)
    circuit.connect(c3.ports[0], v_mid_r)
    circuit.connect(c3.ports[1], GND)

    # C-R-C path of twin-T
    circuit.connect(c1.ports[0], vin)
    circuit.connect(c1.ports[1], v_mid_c)
    circuit.connect(c2.ports[0], v_mid_c)
    circuit.connect(c2.ports[1], vout)
    circuit.connect(r3.ports[0], v_mid_c)
    circuit.connect(r3.ports[1], v_inv)

    # Op-amp
    circuit.connect(u1.ports[0], v_inv)
    circuit.connect(u1.ports[1], GND)
    circuit.connect(u1.ports[2], vout)

    # Feedback
    circuit.connect(r_fb1.ports[0], vout)
    circuit.connect(r_fb1.ports[1], v_inv)

    circuit.connect(r_fb2.ports[0], v_inv)
    circuit.connect(r_fb2.ports[1], GND)

    return circuit, R, C, Q


def main():
    """Demonstrate allpass filters."""
    print("=" * 60)
    print("Allpass Filter (Phase Shifter)")
    print("=" * 60)

    f0 = 1000.0

    circuit1, R, C = build_first_order_allpass(f0)
    circuit2, R2, C2, Q = build_second_order_allpass(f0)

    print(f"""
   Allpass Filter - Constant Magnitude, Variable Phase

   1st Order Allpass:

   Vin ──┬──[R]──┬──[R]──┬─────────┬── Vout
         │       │       │    ┌────┘
         │      [C]      │   [R_fb]
         │       │       │    │
        [R]     GND   [─┬─]───┘
         │            [U1]
        [R]             │
         │             GND
        GND

   Transfer function: H(s) = (s - ω0) / (s + ω0)

   Phase response:
   - f << f0: φ ≈ 0°
   - f = f0:  φ = -90°
   - f >> f0: φ ≈ -180°

   Component values (1st order):
   - R = {R/1000:.2f} kΩ
   - C = {C*1e9:.1f} nF
   - f0 = {f0} Hz

   2nd Order Allpass (Twin-T based):

   Transfer function: H(s) = (s² - s·ω0/Q + ω0²) / (s² + s·ω0/Q + ω0²)

   Phase response:
   - f << f0: φ ≈ 0°
   - f = f0:  φ = -180°
   - f >> f0: φ ≈ -360°

   Applications:
   - Phase equalization in audio
   - Creating phaser effects (cascade multiple)
   - Delay simulation at specific frequencies
   - Notch filter construction (sum with original)
   - Group delay correction
""")

    print("   1st Order Netlist:")
    print(circuit1.build_netlist())

    print("\n   2nd Order Netlist:")
    print(circuit2.build_netlist())

    result1 = circuit1.validate()
    result2 = circuit2.validate()
    print(f"\n   1st Order Validation: {'VALID' if result1.is_valid else 'INVALID'}")
    print(f"   2nd Order Validation: {'VALID' if result2.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
