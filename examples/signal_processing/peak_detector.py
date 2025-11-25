"""Peak Detector (Envelope Detector)

A peak detector tracks the envelope of an AC signal, useful for
AM demodulation, level measurement, and automatic gain control.

Run: python examples/signal_processing/peak_detector.py
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vsin
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def build_simple_peak_detector(
    f_signal: float = 1000.0,
    f_envelope: float = 100.0,
) -> Circuit:
    """Build a simple diode peak detector.

    The RC time constant is chosen to track the envelope while
    filtering the carrier frequency.

    Args:
        f_signal: Carrier/signal frequency
        f_envelope: Expected envelope frequency (for RC sizing)

    Returns:
        Circuit with the peak detector
    """
    circuit = Circuit("simple_peak_detector")

    # RC time constant:
    # - Must be >> 1/f_signal (to smooth carrier ripple)
    # - Must be << 1/f_envelope (to track envelope changes)
    # τ = R * C ≈ 10 / f_signal for good ripple rejection

    C = 100e-9  # 100nF
    R = 10 / (f_signal * C)  # τ = 10 carrier periods

    # Input: AM signal (simplified as sine for demo)
    v_in = Vsin("in", "0 1.0 1000")

    # Diode (fast signal diode)
    d1 = create_component("diode.1n4148", "1")

    # Hold capacitor
    c1 = Capacitor("1", capacitance=C)

    # Discharge resistor
    r1 = Resistor("1", resistance=R)

    circuit.add(v_in, d1, c1, r1)

    # Nets
    vin = Net("vin")
    vout = Net("vout")

    # Input
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)

    # Diode: Vin -> Vout (conducts on positive peaks)
    circuit.connect(d1.ports[0], vin)  # Anode
    circuit.connect(d1.ports[1], vout)  # Cathode

    # Hold capacitor and discharge resistor
    circuit.connect(c1.ports[0], vout)
    circuit.connect(c1.ports[1], GND)
    circuit.connect(r1.ports[0], vout)
    circuit.connect(r1.ports[1], GND)

    return circuit, R, C


def build_precision_peak_detector() -> Circuit:
    """Build a precision peak detector using op-amp.

    Eliminates the diode voltage drop using an op-amp in
    the feedback loop. Provides accurate peak detection
    down to small signal levels.

    Returns:
        Circuit with the precision peak detector
    """
    circuit = Circuit("precision_peak_detector")

    # Input
    v_in = Vsin("in", "0 1.0 1000")

    # Op-amp
    u1 = create_component("opamp.ideal", "1")

    # Diodes (in feedback)
    d1 = create_component("diode.1n4148", "1")  # Charging diode
    d2 = create_component("diode.1n4148", "2")  # Clamp diode

    # Hold capacitor
    c1 = Capacitor("1", capacitance=100e-9)

    # Discharge resistor (high value for slow decay)
    r1 = Resistor("1", resistance=1e6)

    # Buffer resistor
    r2 = Resistor("2", resistance=1000)

    circuit.add(v_in, u1, d1, d2, c1, r1, r2)

    # Nets
    vin = Net("vin")
    vout = Net("vout")
    v_opout = Net("v_opout")

    # Input
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)

    # Op-amp: non-inverting follower configuration
    circuit.connect(u1.ports[0], vout)  # Inverting input (feedback)
    circuit.connect(u1.ports[1], vin)  # Non-inverting input
    circuit.connect(u1.ports[2], v_opout)  # Output

    # D1: Charges capacitor on positive peaks
    circuit.connect(d1.ports[0], v_opout)  # Anode
    circuit.connect(d1.ports[1], vout)  # Cathode

    # D2: Clamps op-amp output when not charging
    circuit.connect(d2.ports[0], vout)  # Anode
    circuit.connect(d2.ports[1], v_opout)  # Cathode

    # Hold capacitor
    circuit.connect(c1.ports[0], vout)
    circuit.connect(c1.ports[1], GND)

    # Discharge resistor
    circuit.connect(r1.ports[0], vout)
    circuit.connect(r1.ports[1], GND)

    # Series resistor (limits charging current)
    circuit.connect(r2.ports[0], v_opout)
    circuit.connect(r2.ports[1], Net("d1_in"))

    return circuit


def build_positive_negative_peak_detector() -> Circuit:
    """Build a detector that captures both positive and negative peaks.

    Uses two peak detectors: one for positive peaks, one for
    negative peaks (inverted). Useful for measuring peak-to-peak
    values.

    Returns:
        Circuit with dual peak detector
    """
    circuit = Circuit("dual_peak_detector")

    # Input
    v_in = Vsin("in", "0 1.0 1000")

    # Positive peak detector
    d_pos = create_component("diode.1n4148", "pos")
    c_pos = Capacitor("pos", capacitance=100e-9)
    r_pos = Resistor("pos", resistance=100_000)

    # Negative peak detector (diode reversed)
    d_neg = create_component("diode.1n4148", "neg")
    c_neg = Capacitor("neg", capacitance=100e-9)
    r_neg = Resistor("neg", resistance=100_000)

    circuit.add(v_in, d_pos, c_pos, r_pos, d_neg, c_neg, r_neg)

    # Nets
    vin = Net("vin")
    v_peak_pos = Net("peak_pos")
    v_peak_neg = Net("peak_neg")

    # Input
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)

    # Positive peak: diode forward from input
    circuit.connect(d_pos.ports[0], vin)
    circuit.connect(d_pos.ports[1], v_peak_pos)
    circuit.connect(c_pos.ports[0], v_peak_pos)
    circuit.connect(c_pos.ports[1], GND)
    circuit.connect(r_pos.ports[0], v_peak_pos)
    circuit.connect(r_pos.ports[1], GND)

    # Negative peak: diode reversed (cathode to input)
    circuit.connect(d_neg.ports[0], v_peak_neg)
    circuit.connect(d_neg.ports[1], vin)
    circuit.connect(c_neg.ports[0], v_peak_neg)
    circuit.connect(c_neg.ports[1], GND)
    circuit.connect(r_neg.ports[0], v_peak_neg)
    circuit.connect(r_neg.ports[1], GND)

    return circuit


def main():
    """Demonstrate peak detector circuits."""
    print("=" * 60)
    print("Peak Detector (Envelope Detector)")
    print("=" * 60)

    circuit1, R, C = build_simple_peak_detector()
    circuit2 = build_precision_peak_detector()
    circuit3 = build_positive_negative_peak_detector()

    print(f"""
   Peak Detector - Envelope Extraction

   1. Simple Diode Peak Detector:

   Vin ────|>|────┬──── Vout (peak)
           D1     │
                 [C]
                  │
                 [R]
                  │
                 GND

   - C charges through D on positive peaks
   - R slowly discharges C between peaks
   - Output follows envelope
   - Error: diode drop (~0.6V)

   Component values:
   - C = {C*1e9:.0f} nF
   - R = {R/1000:.1f} kΩ
   - τ = RC = {R*C*1000:.2f} ms

   2. Precision Peak Detector:

                  ┌──|>|──┐
                  │  D1   │
   Vin ──[+]──[U1]┴──|<|──┼── Vout
          │       D2      │
          └───────────────┤
                         [C]
                          │
                         [R]
                          │
                         GND

   - Op-amp eliminates diode drop
   - D2 prevents op-amp saturation
   - Accurate down to mV levels

   3. Dual Peak Detector (Pos + Neg):

   Vin ──┬──|>|──[C+]──[R+]── V_peak_pos
         │
         └──|<|──[C-]──[R-]── V_peak_neg

   - Captures both positive and negative peaks
   - Vpp = V_peak_pos - V_peak_neg

   Applications:
   - AM demodulation
   - Audio level meters (VU, PPM)
   - Automatic gain control (AGC)
   - Envelope following synthesizers
   - Vibration analysis
""")

    print("   Simple Peak Detector Netlist:")
    print(circuit1.build_netlist())

    result1 = circuit1.validate()
    result2 = circuit2.validate()
    result3 = circuit3.validate()
    print(f"\n   Simple Validation: {'VALID' if result1.is_valid else 'INVALID'}")
    print(f"   Precision Validation: {'VALID' if result2.is_valid else 'INVALID'}")
    print(f"   Dual Validation: {'VALID' if result3.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
