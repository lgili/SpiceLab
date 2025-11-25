"""Instrumentation Amplifier

A three op-amp instrumentation amplifier for precision measurements.

Run: python examples/analog/instrumentation_amp.py
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def build_instrumentation_amp(gain: float = 10) -> Circuit:
    """Build a three op-amp instrumentation amplifier.

    Args:
        gain: Differential gain

    Returns:
        Circuit with the instrumentation amplifier
    """
    circuit = Circuit("instrumentation_amp")

    # Power supplies
    vcc = Vdc("cc", 15.0)
    vee = Vdc("ee", -15.0)

    # Input test voltages
    vin_p = Vdc("in_p", 1.001)  # Slightly different
    vin_n = Vdc("in_n", 1.000)

    # Stage 1 resistors (input buffers with gain)
    # Gain = 1 + 2*R1/Rg
    R1 = 10_000
    Rg = 2 * R1 / (gain - 1) if gain > 1 else 1e9

    r1a = Resistor("1a", resistance=R1)
    r1b = Resistor("1b", resistance=R1)
    rg = Resistor("g", resistance=Rg)

    # Stage 2 resistors (differential to single-ended)
    R2 = 10_000
    r2a = Resistor("2a", resistance=R2)
    r2b = Resistor("2b", resistance=R2)
    r3a = Resistor("3a", resistance=R2)
    r3b = Resistor("3b", resistance=R2)

    # Three op-amps
    u1 = create_component("opamp.tl072", "1")
    u2 = create_component("opamp.tl072", "2")
    u3 = create_component("opamp.tl072", "3")

    circuit.add(vcc, vee, vin_p, vin_n, r1a, r1b, rg, r2a, r2b, r3a, r3b, u1, u2, u3)

    # Nets
    vcc_net = Net("vcc")
    vee_net = Net("vee")
    vinp_net = Net("vinp")
    vinn_net = Net("vinn")
    v1out = Net("v1out")
    v2out = Net("v2out")
    v1fb = Net("v1fb")
    v2fb = Net("v2fb")
    v3minus = Net("v3minus")
    vout = Net("vout")

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

    # U1: Top input buffer (out, +, -, V+, V-)
    circuit.connect(u1.ports[0], v1out)
    circuit.connect(u1.ports[1], vinp_net)
    circuit.connect(u1.ports[2], v1fb)
    circuit.connect(u1.ports[3], vcc_net)
    circuit.connect(u1.ports[4], vee_net)

    # U2: Bottom input buffer
    circuit.connect(u2.ports[0], v2out)
    circuit.connect(u2.ports[1], vinn_net)
    circuit.connect(u2.ports[2], v2fb)
    circuit.connect(u2.ports[3], vcc_net)
    circuit.connect(u2.ports[4], vee_net)

    # Stage 1 feedback: R1a, Rg, R1b chain
    circuit.connect(r1a.ports[0], v1out)
    circuit.connect(r1a.ports[1], v1fb)
    circuit.connect(rg.ports[0], v1fb)
    circuit.connect(rg.ports[1], v2fb)
    circuit.connect(r1b.ports[0], v2fb)
    circuit.connect(r1b.ports[1], v2out)

    # Stage 2: Difference amplifier
    circuit.connect(r2a.ports[0], v1out)
    circuit.connect(r2a.ports[1], v3minus)
    circuit.connect(r3a.ports[0], v3minus)
    circuit.connect(r3a.ports[1], vout)

    circuit.connect(r2b.ports[0], v2out)
    circuit.connect(r2b.ports[1], Net("v3plus"))
    circuit.connect(r3b.ports[0], Net("v3plus"))
    circuit.connect(r3b.ports[1], GND)

    # U3: Output difference amplifier
    circuit.connect(u3.ports[0], vout)
    circuit.connect(u3.ports[1], Net("v3plus"))
    circuit.connect(u3.ports[2], v3minus)
    circuit.connect(u3.ports[3], vcc_net)
    circuit.connect(u3.ports[4], vee_net)

    return circuit, R1, Rg, R2


def main():
    """Demonstrate instrumentation amplifier."""
    print("=" * 60)
    print("Instrumentation Amplifier")
    print("=" * 60)

    gain = 10
    circuit, R1, Rg, R2 = build_instrumentation_amp(gain)

    vin_diff = 1.001 - 1.000
    vout_expected = vin_diff * gain

    print(f"""
   Three Op-Amp Instrumentation Amplifier

   Circuit topology:

   Vin+ ──(+)U1─┬──[R2a]──┬──(-)U3──┐
                │         │         │
               [R1a]     [R3a]     [OP]── Vout
                │         │         │
               [Rg]      [R3b]    (+)
                │         │         │
               [R1b]     GND    ┌──┘
                │               │
   Vin- ──(+)U2─┴──[R2b]────────┘

   Design:
   - Gain = 1 + 2×R1/Rg = 1 + 2×{R1}/{Rg:.0f} = {gain}
   - R1 = R1a = R1b = {R1} Ω
   - Rg = {Rg:.0f} Ω
   - R2 = R3 = {R2} Ω

   Test signals:
   - Vin+ = 1.001V, Vin- = 1.000V
   - Differential input: {vin_diff*1000:.1f} mV
   - Expected output: {vout_expected*1000:.1f} mV

   Advantages:
   - Very high input impedance
   - Excellent CMRR (>100dB possible)
   - Single resistor (Rg) gain adjustment
   - Balanced inputs
""")

    print("   Netlist:")
    print(circuit.build_netlist())

    result = circuit.validate()
    print(f"\n   Validation: {'VALID' if result.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
