"""RF Amplifier with Matching Networks

RF amplifiers require careful impedance matching for maximum power
transfer and stability. This example shows a common-emitter RF amp
with input and output matching networks.

Run: python examples/rf/rf_amplifier.py
"""

import math

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Inductor, Resistor, Vac
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def build_matched_rf_amplifier(
    f_center: float = 100e6,
    z_source: float = 50,
    z_load: float = 50,
) -> Circuit:
    """Build an RF amplifier with L-match networks.

    Uses L-section matching networks at input and output to
    transform 50Ω source/load to transistor impedances.

    Args:
        f_center: Center frequency in Hz
        z_source: Source impedance (typically 50Ω)
        z_load: Load impedance (typically 50Ω)

    Returns:
        Circuit with matched RF amplifier
    """
    circuit = Circuit("matched_rf_amplifier")

    omega = 2 * math.pi * f_center

    # Transistor (2N3904 for simplicity, real RF uses 2N5179 etc.)
    q1 = create_component("bjt.2n3904", "1")

    # Assume transistor input Z ≈ 1kΩ, output Z ≈ 10kΩ at RF
    # (Real values from datasheet/measurement)
    z_in_transistor = 1000
    z_out_transistor = 10000

    # Input L-match: 50Ω to ~1kΩ
    # Q = √(Rhigh/Rlow - 1)
    Q_in = math.sqrt(z_in_transistor / z_source - 1)
    # Xs = Q × Rlow, Xp = Rhigh / Q
    Xs_in = Q_in * z_source
    Xp_in = z_in_transistor / Q_in

    L_in = Xs_in / omega  # Series inductor
    C_in = 1 / (omega * Xp_in)  # Shunt capacitor

    # Output L-match: ~10kΩ to 50Ω
    Q_out = math.sqrt(z_out_transistor / z_load - 1)
    Xp_out = z_out_transistor / Q_out
    Xs_out = Q_out * z_load

    C_out_shunt = 1 / (omega * Xp_out)  # Shunt capacitor (at collector)
    L_out = Xs_out / omega  # Series inductor

    # Input matching components
    l_in = Inductor("in", inductance=L_in)
    c_in_match = Capacitor("in_match", capacitance=C_in)

    # Output matching components
    c_out_match = Capacitor("out_match", capacitance=C_out_shunt)
    l_out = Inductor("out", inductance=L_out)

    # Bias components
    r_b1 = Resistor("b1", resistance=22_000)
    r_b2 = Resistor("b2", resistance=4700)
    r_e = Resistor("e", resistance=470)
    c_e = Capacitor("e", capacitance=1e-9)  # RF bypass

    # RF choke for collector bias
    l_rfc = Inductor("rfc", inductance=1e-6)

    # DC blocking capacitors
    c_dc_in = Capacitor("dc_in", capacitance=100e-12)
    c_dc_out = Capacitor("dc_out", capacitance=100e-12)

    # Input source
    v_in = Vac("in", ac_mag=0.01)  # 10mV RF input

    # Power supply
    v_cc = create_component("source.vdc", "cc")

    circuit.add(q1, l_in, c_in_match, c_out_match, l_out)
    circuit.add(r_b1, r_b2, r_e, c_e, l_rfc, c_dc_in, c_dc_out)
    circuit.add(v_in, v_cc)

    # Nets
    vcc = Net("vcc")
    vin = Net("vin")
    v_match_in = Net("v_match_in")
    v_base = Net("v_base")
    v_coll = Net("v_coll")
    v_emit = Net("v_emit")
    v_match_out = Net("v_match_out")
    vout = Net("vout")

    # Power supply
    circuit.connect(v_cc.ports[0], vcc)
    circuit.connect(v_cc.ports[1], GND)

    # Input source
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)

    # Input matching: series L, shunt C
    circuit.connect(l_in.ports[0], vin)
    circuit.connect(l_in.ports[1], v_match_in)
    circuit.connect(c_in_match.ports[0], v_match_in)
    circuit.connect(c_in_match.ports[1], GND)

    # DC blocking to base
    circuit.connect(c_dc_in.ports[0], v_match_in)
    circuit.connect(c_dc_in.ports[1], v_base)

    # Bias network
    circuit.connect(r_b1.ports[0], vcc)
    circuit.connect(r_b1.ports[1], v_base)
    circuit.connect(r_b2.ports[0], v_base)
    circuit.connect(r_b2.ports[1], GND)

    # Transistor
    circuit.connect(q1.ports[0], v_coll)
    circuit.connect(q1.ports[1], v_base)
    circuit.connect(q1.ports[2], v_emit)

    # Emitter resistor with RF bypass
    circuit.connect(r_e.ports[0], v_emit)
    circuit.connect(r_e.ports[1], GND)
    circuit.connect(c_e.ports[0], v_emit)
    circuit.connect(c_e.ports[1], GND)

    # Collector bias through RFC
    circuit.connect(l_rfc.ports[0], vcc)
    circuit.connect(l_rfc.ports[1], v_coll)

    # Output matching: shunt C at collector, series L to output
    circuit.connect(c_out_match.ports[0], v_coll)
    circuit.connect(c_out_match.ports[1], GND)

    circuit.connect(c_dc_out.ports[0], v_coll)
    circuit.connect(c_dc_out.ports[1], v_match_out)

    circuit.connect(l_out.ports[0], v_match_out)
    circuit.connect(l_out.ports[1], vout)

    return circuit, L_in, C_in, L_out, C_out_shunt, Q_in, Q_out


def build_cascode_rf_amplifier(f_center: float = 100e6) -> Circuit:
    """Build a cascode RF amplifier.

    The cascode configuration (CE + CB) provides high gain,
    wide bandwidth, and good isolation. Reduces Miller effect.

    Args:
        f_center: Center frequency in Hz

    Returns:
        Circuit with cascode RF amplifier
    """
    circuit = Circuit("cascode_rf_amplifier")

    # Two transistors
    q1 = create_component("bjt.2n3904", "1")  # Common emitter
    q2 = create_component("bjt.2n3904", "2")  # Common base

    # Bias resistors
    r_b1 = Resistor("b1", resistance=22_000)  # Q1 base bias
    r_b2 = Resistor("b2", resistance=4700)
    r_cb = Resistor("cb", resistance=10_000)  # Q2 base bias
    r_e = Resistor("e", resistance=470)
    r_c = Resistor("c", resistance=1000)

    # RF bypass capacitors
    c_e = Capacitor("e", capacitance=1e-9)
    c_cb = Capacitor("cb", capacitance=100e-9)  # Q2 base bypass

    # Coupling capacitors
    c_in = Capacitor("in", capacitance=100e-12)
    c_out = Capacitor("out", capacitance=100e-12)

    # Input source
    v_in = Vac("in", ac_mag=0.01)

    # Power supply
    v_cc = create_component("source.vdc", "cc")

    # Q2 base bias voltage
    v_bias = create_component("source.vdc", "bias")

    circuit.add(q1, q2, r_b1, r_b2, r_cb, r_e, r_c, c_e, c_cb, c_in, c_out)
    circuit.add(v_in, v_cc, v_bias)

    # Nets
    vcc = Net("vcc")
    vin = Net("vin")
    v_b1 = Net("v_b1")  # Q1 base
    v_e1 = Net("v_e1")  # Q1 emitter
    v_c1 = Net("v_c1")  # Q1 collector = Q2 emitter
    v_b2 = Net("v_b2")  # Q2 base (fixed bias)
    v_c2 = Net("v_c2")  # Q2 collector
    vout = Net("vout")
    v_bias_net = Net("v_bias")

    # Power supply
    circuit.connect(v_cc.ports[0], vcc)
    circuit.connect(v_cc.ports[1], GND)

    # Q2 base bias (typically ~Vcc/2)
    circuit.connect(v_bias.ports[0], v_bias_net)
    circuit.connect(v_bias.ports[1], GND)

    # Input
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)

    # Input coupling
    circuit.connect(c_in.ports[0], vin)
    circuit.connect(c_in.ports[1], v_b1)

    # Q1 bias
    circuit.connect(r_b1.ports[0], vcc)
    circuit.connect(r_b1.ports[1], v_b1)
    circuit.connect(r_b2.ports[0], v_b1)
    circuit.connect(r_b2.ports[1], GND)

    # Q1 (common emitter)
    circuit.connect(q1.ports[0], v_c1)  # Collector
    circuit.connect(q1.ports[1], v_b1)  # Base
    circuit.connect(q1.ports[2], v_e1)  # Emitter

    # Q1 emitter resistor and bypass
    circuit.connect(r_e.ports[0], v_e1)
    circuit.connect(r_e.ports[1], GND)
    circuit.connect(c_e.ports[0], v_e1)
    circuit.connect(c_e.ports[1], GND)

    # Q2 base bias
    circuit.connect(r_cb.ports[0], v_bias_net)
    circuit.connect(r_cb.ports[1], v_b2)
    circuit.connect(c_cb.ports[0], v_b2)
    circuit.connect(c_cb.ports[1], GND)

    # Q2 (common base) - emitter driven by Q1 collector
    circuit.connect(q2.ports[0], v_c2)  # Collector
    circuit.connect(q2.ports[1], v_b2)  # Base
    circuit.connect(q2.ports[2], v_c1)  # Emitter

    # Collector resistor
    circuit.connect(r_c.ports[0], vcc)
    circuit.connect(r_c.ports[1], v_c2)

    # Output coupling
    circuit.connect(c_out.ports[0], v_c2)
    circuit.connect(c_out.ports[1], vout)

    return circuit


def build_feedback_rf_amplifier() -> Circuit:
    """Build a broadband feedback RF amplifier.

    Negative feedback provides flat gain over wide bandwidth
    at the expense of gain. Good for wideband applications.

    Returns:
        Circuit with feedback RF amplifier
    """
    circuit = Circuit("feedback_rf_amplifier")

    # Transistor
    q1 = create_component("bjt.2n3904", "1")

    # Feedback resistor (collector to base)
    # Sets gain ≈ Rf / Re (approximately)
    r_f = Resistor("f", resistance=2200)

    # Emitter resistor (partially bypassed for RF)
    r_e = Resistor("e", resistance=100)
    r_e2 = Resistor("e2", resistance=10)  # Unbypassed portion
    c_e = Capacitor("e", capacitance=10e-9)

    # Collector resistor
    r_c = Resistor("c", resistance=470)

    # Coupling capacitors
    c_in = Capacitor("in", capacitance=100e-12)
    c_out = Capacitor("out", capacitance=100e-12)

    # Input source
    v_in = Vac("in", ac_mag=0.01)

    # Power supply
    v_cc = create_component("source.vdc", "cc")

    circuit.add(q1, r_f, r_e, r_e2, c_e, r_c, c_in, c_out, v_in, v_cc)

    # Nets
    vcc = Net("vcc")
    vin = Net("vin")
    v_base = Net("v_base")
    v_coll = Net("v_coll")
    v_e1 = Net("v_e1")
    vout = Net("vout")

    # Power supply
    circuit.connect(v_cc.ports[0], vcc)
    circuit.connect(v_cc.ports[1], GND)

    # Input
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)

    # Input coupling
    circuit.connect(c_in.ports[0], vin)
    circuit.connect(c_in.ports[1], v_base)

    # Transistor
    circuit.connect(q1.ports[0], v_coll)
    circuit.connect(q1.ports[1], v_base)
    circuit.connect(q1.ports[2], v_e1)

    # Feedback resistor (collector to base)
    circuit.connect(r_f.ports[0], v_coll)
    circuit.connect(r_f.ports[1], v_base)

    # Emitter network: Re2 always in circuit, Re+Ce in parallel
    circuit.connect(r_e2.ports[0], v_e1)
    circuit.connect(r_e2.ports[1], GND)

    circuit.connect(r_e.ports[0], v_e1)
    circuit.connect(r_e.ports[1], GND)
    circuit.connect(c_e.ports[0], v_e1)
    circuit.connect(c_e.ports[1], GND)

    # Collector resistor
    circuit.connect(r_c.ports[0], vcc)
    circuit.connect(r_c.ports[1], v_coll)

    # Output coupling
    circuit.connect(c_out.ports[0], v_coll)
    circuit.connect(c_out.ports[1], vout)

    return circuit


def main():
    """Demonstrate RF amplifier circuits."""
    print("=" * 60)
    print("RF Amplifier with Matching Networks")
    print("=" * 60)

    f_center = 100e6  # 100 MHz

    circuit1, L_in, C_in, L_out, C_out, Q_in, Q_out = build_matched_rf_amplifier(f_center)
    circuit2 = build_cascode_rf_amplifier(f_center)
    circuit3 = build_feedback_rf_amplifier()

    print(f"""
   RF Amplifiers for {f_center/1e6:.0f} MHz

   1. Matched Common-Emitter Amplifier:

   50Ω ──[L_in]──┬──||──[Q]──┬──||──[L_out]── 50Ω
                 │      │     │
               [C_in]  bias [C_out]
                 │           │
                GND         GND

   Input L-Match (50Ω → ~1kΩ):
   - Series L = {L_in*1e9:.1f} nH
   - Shunt C = {C_in*1e12:.1f} pF
   - Q = {Q_in:.1f}

   Output L-Match (~10kΩ → 50Ω):
   - Shunt C = {C_out*1e12:.1f} pF
   - Series L = {L_out*1e9:.1f} nH
   - Q = {Q_out:.1f}

   2. Cascode Amplifier:

   Vin ──||──[Q1:CE]──[Q2:CB]──||── Vout
               │         │
              bias      bias

   Advantages:
   - High gain (CE) × isolation (CB)
   - Wide bandwidth (reduced Miller)
   - High output impedance
   - Good reverse isolation

   3. Feedback Amplifier:

            ┌────[Rf]────┐
            │            │
   Vin ──||─┴──[Q]───────┼──||── Vout
                │        │
               [Re]     [Rc]
                │        │
               GND      Vcc

   Advantages:
   - Very wide bandwidth
   - Flat frequency response
   - Good input/output match
   - Predictable gain ≈ Rf/Re

   RF Amplifier Parameters:
   ┌────────────────┬─────────────────────────┐
   │ Gain           │ dB power/voltage gain   │
   │ Bandwidth      │ -3dB frequency range    │
   │ Noise Figure   │ Added noise (dB)        │
   │ IP3 (IIP3)     │ Third-order intercept   │
   │ P1dB           │ 1dB compression point   │
   │ Return Loss    │ Match quality (S11,S22) │
   │ Isolation      │ Reverse gain (S12)      │
   │ Stability      │ K-factor > 1            │
   └────────────────┴─────────────────────────┘
""")

    print("   Matched Amplifier Netlist:")
    print(circuit1.build_netlist())

    result1 = circuit1.validate()
    result2 = circuit2.validate()
    result3 = circuit3.validate()
    print(f"\n   Matched Amp Validation: {'VALID' if result1.is_valid else 'INVALID'}")
    print(f"   Cascode Validation: {'VALID' if result2.is_valid else 'INVALID'}")
    print(f"   Feedback Validation: {'VALID' if result3.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
