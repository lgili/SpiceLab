"""Automatic Gain Control (AGC)

AGC circuits automatically adjust gain to maintain constant output
level regardless of input signal amplitude. Essential for radio
receivers, audio systems, and instrumentation.

Run: python examples/signal_processing/agc_circuit.py
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vsin
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def build_simple_agc() -> Circuit:
    """Build a simple AGC using JFET as voltage-controlled resistor.

    The JFET's drain-source resistance varies with gate voltage,
    providing variable attenuation. A peak detector generates
    the control voltage.

    Returns:
        Circuit with simple AGC
    """
    circuit = Circuit("simple_agc")

    # Input signal (variable amplitude for AGC demo)
    v_in = Vsin("in", "0 1.0 1000")

    # JFET as variable resistor
    j1 = create_component("jfet.2n5457", "1")

    # Op-amp for gain stage
    u1 = create_component("opamp.ideal", "1")

    # Peak detector components
    d1 = create_component("diode.1n4148", "1")
    c_hold = Capacitor("hold", capacitance=10e-6)  # Hold capacitor
    r_discharge = Resistor("dis", resistance=100_000)  # Slow discharge

    # Gain setting resistors
    r_in = Resistor("in", resistance=10_000)
    r_fb = Resistor("fb", resistance=100_000)  # Max gain = 10

    # Input coupling
    c_in = Capacitor("in", capacitance=1e-6)

    circuit.add(v_in, j1, u1, d1, c_hold, r_discharge, r_in, r_fb, c_in)

    # Nets
    vin = Net("vin")
    v_coupled = Net("v_coupled")
    v_atten = Net("v_atten")
    vout = Net("vout")
    v_inv = Net("v_inv")
    v_ctrl = Net("v_ctrl")

    # Input coupling
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)
    circuit.connect(c_in.ports[0], vin)
    circuit.connect(c_in.ports[1], v_coupled)

    # JFET as voltage divider (D-S channel)
    circuit.connect(j1.ports[0], v_coupled)  # Drain
    circuit.connect(j1.ports[1], v_ctrl)  # Gate (control voltage)
    circuit.connect(j1.ports[2], v_atten)  # Source

    # Load resistor for JFET divider
    r_load = Resistor("load", resistance=10_000)
    circuit.add(r_load)
    circuit.connect(r_load.ports[0], v_atten)
    circuit.connect(r_load.ports[1], GND)

    # Amplifier stage
    circuit.connect(r_in.ports[0], v_atten)
    circuit.connect(r_in.ports[1], v_inv)

    circuit.connect(u1.ports[0], v_inv)
    circuit.connect(u1.ports[1], GND)
    circuit.connect(u1.ports[2], vout)

    circuit.connect(r_fb.ports[0], v_inv)
    circuit.connect(r_fb.ports[1], vout)

    # Peak detector: Output -> Control voltage
    circuit.connect(d1.ports[0], vout)
    circuit.connect(d1.ports[1], v_ctrl)

    circuit.connect(c_hold.ports[0], v_ctrl)
    circuit.connect(c_hold.ports[1], GND)

    circuit.connect(r_discharge.ports[0], v_ctrl)
    circuit.connect(r_discharge.ports[1], GND)

    return circuit


def build_vca_agc() -> Circuit:
    """Build AGC using a voltage-controlled amplifier concept.

    Uses an analog multiplier/VCA topology where gain is
    controlled by a DC voltage derived from the output level.

    Returns:
        Circuit with VCA-based AGC
    """
    circuit = Circuit("vca_agc")

    # Input
    v_in = Vsin("in", "0 1.0 1000")

    # Differential pair for variable gain
    q1 = create_component("bjt.2n3904", "1")
    q2 = create_component("bjt.2n3904", "2")

    # Current source (tail)
    r_tail = Resistor("tail", resistance=10_000)

    # Collector resistors
    r_c1 = Resistor("c1", resistance=10_000)
    r_c2 = Resistor("c2", resistance=10_000)

    # Emitter degeneration for linearity
    r_e1 = Resistor("e1", resistance=100)
    r_e2 = Resistor("e2", resistance=100)

    # Control voltage source (from peak detector)
    v_ctrl = Vsin("ctrl", "2.5 0 0")  # DC bias for control

    # Peak detector
    d1 = create_component("diode.1n4148", "1")
    c_peak = Capacitor("peak", capacitance=10e-6)
    r_peak = Resistor("peak", resistance=100_000)

    # Input coupling
    c_in = Capacitor("in", capacitance=1e-6)

    circuit.add(v_in, q1, q2, r_tail, r_c1, r_c2, r_e1, r_e2, v_ctrl)
    circuit.add(d1, c_peak, r_peak, c_in)

    # Supply
    v_cc = create_component("source.vdc", "cc")
    circuit.add(v_cc)

    # Nets
    vin = Net("vin")
    v_coupled = Net("v_coupled")
    vcc = Net("vcc")
    v_tail = Net("v_tail")
    vout_p = Net("vout_p")
    vout_n = Net("vout_n")
    v_e1 = Net("v_e1")
    v_e2 = Net("v_e2")
    v_control = Net("v_control")

    # Power supply
    circuit.connect(v_cc.ports[0], vcc)
    circuit.connect(v_cc.ports[1], GND)

    # Input coupling
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)
    circuit.connect(c_in.ports[0], vin)
    circuit.connect(c_in.ports[1], v_coupled)

    # Differential pair
    circuit.connect(q1.ports[0], vout_n)  # Collector
    circuit.connect(q1.ports[1], v_coupled)  # Base (signal)
    circuit.connect(q1.ports[2], v_e1)  # Emitter

    circuit.connect(q2.ports[0], vout_p)  # Collector
    circuit.connect(q2.ports[1], v_control)  # Base (control)
    circuit.connect(q2.ports[2], v_e2)  # Emitter

    # Emitter resistors
    circuit.connect(r_e1.ports[0], v_e1)
    circuit.connect(r_e1.ports[1], v_tail)
    circuit.connect(r_e2.ports[0], v_e2)
    circuit.connect(r_e2.ports[1], v_tail)

    # Tail resistor (current source)
    circuit.connect(r_tail.ports[0], v_tail)
    circuit.connect(r_tail.ports[1], GND)

    # Collector resistors
    circuit.connect(r_c1.ports[0], vcc)
    circuit.connect(r_c1.ports[1], vout_n)
    circuit.connect(r_c2.ports[0], vcc)
    circuit.connect(r_c2.ports[1], vout_p)

    # Control voltage
    circuit.connect(v_ctrl.ports[0], v_control)
    circuit.connect(v_ctrl.ports[1], GND)

    # Peak detector from output
    circuit.connect(d1.ports[0], vout_p)
    circuit.connect(d1.ports[1], Net("peak_out"))
    circuit.connect(c_peak.ports[0], Net("peak_out"))
    circuit.connect(c_peak.ports[1], GND)
    circuit.connect(r_peak.ports[0], Net("peak_out"))
    circuit.connect(r_peak.ports[1], GND)

    return circuit


def build_feed_forward_agc() -> Circuit:
    """Build a feed-forward AGC.

    Measures input level before amplification and sets gain
    accordingly. Faster response than feedback AGC but less
    accurate.

    Returns:
        Circuit with feed-forward AGC
    """
    circuit = Circuit("feedforward_agc")

    # Input
    v_in = Vsin("in", "0 1.0 1000")

    # Level detector op-amp (rectifier + filter)
    u1 = create_component("opamp.ideal", "1")
    d1 = create_component("diode.1n4148", "1")

    # Main amplifier
    u2 = create_component("opamp.ideal", "2")

    # JFET for gain control
    j1 = create_component("jfet.2n5457", "1")

    # Level detector components
    r_det1 = Resistor("det1", resistance=10_000)
    r_det2 = Resistor("det2", resistance=10_000)
    c_filt = Capacitor("filt", capacitance=10e-6)

    # Amplifier components
    r_in = Resistor("in", resistance=10_000)
    r_fb = Resistor("fb", resistance=100_000)

    circuit.add(v_in, u1, u2, d1, j1, r_det1, r_det2, c_filt, r_in, r_fb)

    # Nets
    vin = Net("vin")
    vout = Net("vout")
    v_level = Net("v_level")
    v_ctrl = Net("v_ctrl")
    v_inv1 = Net("v_inv1")
    v_inv2 = Net("v_inv2")

    # Input
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)

    # Level detector: rectify input
    circuit.connect(r_det1.ports[0], vin)
    circuit.connect(r_det1.ports[1], v_inv1)

    circuit.connect(u1.ports[0], v_inv1)
    circuit.connect(u1.ports[1], GND)
    circuit.connect(u1.ports[2], v_level)

    circuit.connect(d1.ports[0], v_level)
    circuit.connect(d1.ports[1], v_ctrl)

    circuit.connect(r_det2.ports[0], v_ctrl)
    circuit.connect(r_det2.ports[1], v_inv1)

    circuit.connect(c_filt.ports[0], v_ctrl)
    circuit.connect(c_filt.ports[1], GND)

    # Main amplifier with JFET-controlled input attenuation
    circuit.connect(j1.ports[0], vin)  # Drain
    circuit.connect(j1.ports[1], v_ctrl)  # Gate
    circuit.connect(j1.ports[2], Net("v_atten"))  # Source

    r_load = Resistor("load", resistance=10_000)
    circuit.add(r_load)
    circuit.connect(r_load.ports[0], Net("v_atten"))
    circuit.connect(r_load.ports[1], GND)

    circuit.connect(r_in.ports[0], Net("v_atten"))
    circuit.connect(r_in.ports[1], v_inv2)

    circuit.connect(u2.ports[0], v_inv2)
    circuit.connect(u2.ports[1], GND)
    circuit.connect(u2.ports[2], vout)

    circuit.connect(r_fb.ports[0], v_inv2)
    circuit.connect(r_fb.ports[1], vout)

    return circuit


def main():
    """Demonstrate AGC circuits."""
    print("=" * 60)
    print("Automatic Gain Control (AGC)")
    print("=" * 60)

    circuit1 = build_simple_agc()
    circuit2 = build_vca_agc()
    circuit3 = build_feed_forward_agc()

    print("""
   Automatic Gain Control - Constant Output Level

   1. Simple JFET AGC:

   Vin ──||──[JFET]──┬──[Rin]──┬──[Rfb]── Vout
              │      │         │    │
              G      [R]      [─]   │
              │       │       [U1]──┘
              └──|>|──┴──[C]──GND

   Operation:
   - JFET acts as voltage-controlled resistor
   - Peak detector creates control voltage from output
   - High output → negative Vgs → high JFET resistance → lower gain
   - Self-regulating feedback loop

   2. VCA-Based AGC:

   Uses differential pair where one transistor gets signal,
   other gets control voltage. Gain proportional to control.

   Advantages:
   - Wide dynamic range
   - Good linearity
   - Fast response

   3. Feed-Forward AGC:

   Vin ──┬──[Level Det]──[Control]
         │                  │
         └──[JFET]──[Amp]──┴── Vout

   - Measures input level BEFORE amplification
   - Sets gain based on input, not output
   - Faster attack time
   - Less accurate than feedback AGC

   Key Parameters:
   - Attack time: How fast gain reduces for loud signals
   - Release time: How fast gain increases after loud signal
   - Compression ratio: dB input change per dB output change
   - Threshold: Level where AGC begins acting

   Applications:
   - Radio receivers (AM/FM/SSB)
   - Audio compressors/limiters
   - Hearing aids
   - Instrumentation amplifiers
   - Telecommunications
""")

    print("   Simple AGC Netlist:")
    print(circuit1.build_netlist())

    result1 = circuit1.validate()
    result2 = circuit2.validate()
    result3 = circuit3.validate()
    print(f"\n   Simple AGC Validation: {'VALID' if result1.is_valid else 'INVALID'}")
    print(f"   VCA AGC Validation: {'VALID' if result2.is_valid else 'INVALID'}")
    print(f"   Feed-Forward Validation: {'VALID' if result3.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
