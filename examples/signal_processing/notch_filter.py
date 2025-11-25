"""Twin-T Notch Filter

A notch filter for rejecting a specific frequency (e.g., 50/60Hz mains hum).
The Twin-T topology provides deep notch with simple passive components.

Run: python examples/signal_processing/notch_filter.py
"""

import math

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vac
from spicelab.core.net import GND, Net


def build_twin_t_notch(f_notch: float = 60.0, impedance: float = 10_000) -> Circuit:
    """Build a Twin-T notch filter.

    The Twin-T uses two T-networks in parallel:
    - R-C-R T network (highpass characteristic)
    - C-R-C T network (lowpass characteristic)

    At f_notch, the two paths cancel, creating a deep null.

    Args:
        f_notch: Notch frequency in Hz
        impedance: Base impedance (R value)

    Returns:
        Circuit with the Twin-T notch filter
    """
    # Component values for Twin-T:
    # f_notch = 1 / (2 * pi * R * C)
    # C = 1 / (2 * pi * R * f_notch)
    R = impedance
    C = 1 / (2 * math.pi * R * f_notch)

    circuit = Circuit("twin_t_notch")

    # Input source
    v_in = Vac("in", ac_mag=1.0)

    # R-C-R T network (top path)
    r1 = Resistor("1", resistance=R)
    r2 = Resistor("2", resistance=R)
    c1 = Capacitor("1", capacitance=2 * C)  # 2C to ground

    # C-R-C T network (bottom path)
    c2 = Capacitor("2", capacitance=C)
    c3 = Capacitor("3", capacitance=C)
    r3 = Resistor("3", resistance=R / 2)  # R/2 to ground

    circuit.add(v_in, r1, r2, c1, c2, c3, r3)

    # Nets
    vin = Net("vin")
    vout = Net("vout")
    v_mid_r = Net("mid_r")  # Middle of R-C-R path
    v_mid_c = Net("mid_c")  # Middle of C-R-C path

    # Input source
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)

    # R-C-R path: Vin -> R1 -> mid_r -> R2 -> Vout
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], v_mid_r)
    circuit.connect(r2.ports[0], v_mid_r)
    circuit.connect(r2.ports[1], vout)
    circuit.connect(c1.ports[0], v_mid_r)
    circuit.connect(c1.ports[1], GND)

    # C-R-C path: Vin -> C2 -> mid_c -> C3 -> Vout
    circuit.connect(c2.ports[0], vin)
    circuit.connect(c2.ports[1], v_mid_c)
    circuit.connect(c3.ports[0], v_mid_c)
    circuit.connect(c3.ports[1], vout)
    circuit.connect(r3.ports[0], v_mid_c)
    circuit.connect(r3.ports[1], GND)

    return circuit, R, C


def main():
    """Demonstrate Twin-T notch filter."""
    print("=" * 60)
    print("Twin-T Notch Filter")
    print("=" * 60)

    f_notch = 60.0  # 60Hz mains rejection
    impedance = 10_000

    circuit, R, C = build_twin_t_notch(f_notch, impedance)

    print(f"""
   Twin-T Notch Filter for {f_notch} Hz rejection

   Circuit topology:

                R1          R2
   Vin ────┬────[R]────┬────[R]────┬──── Vout
           │           │           │
           │          [2C]         │
           │           │           │
           │          GND          │
           │                       │
           └────[C]────┬────[C]────┘
                       │
                     [R/2]
                       │
                      GND

   Design equations:
   - f_notch = 1 / (2π × R × C)
   - For symmetric null: R1 = R2 = R, C2 = C3 = C
   - Center elements: 2C and R/2

   Component values:
   - R1 = R2 = {R/1000:.1f} kΩ
   - C2 = C3 = {C*1e9:.2f} nF
   - C1 (center) = {2*C*1e9:.2f} nF
   - R3 (center) = {R/2/1000:.2f} kΩ

   Characteristics:
   - Notch frequency: {f_notch} Hz
   - Theoretical notch depth: -40 to -60 dB (ideal)
   - Q factor: ~0.25 (passive) - can be increased with feedback
   - Passband gain: 0 dB (unity)

   Applications:
   - 50/60 Hz mains hum rejection
   - Audio equipment noise filtering
   - Instrumentation signal conditioning
   - Biomedical signal processing (ECG/EEG)
""")

    print("   Netlist:")
    print(circuit.build_netlist())

    result = circuit.validate()
    print(f"\n   Validation: {'VALID' if result.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
