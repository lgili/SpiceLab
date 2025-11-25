"""Tutorial 08: Operational Amplifiers

This tutorial teaches you how to use op-amps:
1. Non-inverting amplifier
2. Inverting amplifier
3. Voltage follower (buffer)
4. Summing amplifier

Run: python examples/tutorials/08_opamps.py
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.library import create_component, get_component_spec


def demo_non_inverting():
    """Demonstrate non-inverting amplifier."""
    print("\n1. Non-Inverting Amplifier")
    print("-" * 40)

    circuit = Circuit("non_inverting_amp")

    # Power supplies
    vcc = Vdc("cc", 15.0)
    vee = Vdc("ee", -15.0)
    vin = Vdc("in", 1.0)

    # Components
    r1 = Resistor("1", resistance=10_000)
    rf = Resistor("f", resistance=100_000)
    opamp = create_component("opamp.lm741", "1")

    circuit.add(vcc, vee, vin, r1, rf, opamp)

    # Nets
    vcc_net = Net("vcc")
    vee_net = Net("vee")
    vin_net = Net("vin")
    vout = Net("vout")
    vminus = Net("vminus")

    # Power supplies (note: vee negative terminal is positive)
    circuit.connect(vcc.ports[0], vcc_net)
    circuit.connect(vcc.ports[1], GND)
    circuit.connect(vee.ports[0], GND)
    circuit.connect(vee.ports[1], vee_net)

    # Input signal
    circuit.connect(vin.ports[0], vin_net)
    circuit.connect(vin.ports[1], GND)

    # Op-amp: out, +in, -in, V+, V-
    circuit.connect(opamp.ports[0], vout)
    circuit.connect(opamp.ports[1], vin_net)  # + input
    circuit.connect(opamp.ports[2], vminus)  # - input
    circuit.connect(opamp.ports[3], vcc_net)
    circuit.connect(opamp.ports[4], vee_net)

    # Feedback network
    circuit.connect(rf.ports[0], vout)
    circuit.connect(rf.ports[1], vminus)
    circuit.connect(r1.ports[0], vminus)
    circuit.connect(r1.ports[1], GND)

    gain = 1 + 100_000 / 10_000

    print(f"""
   Non-Inverting Amplifier

   Circuit:
                  ┌───[Rf]───┐
                  │          │
   Vin ──(+)──────┤−         │
                  │    741   ├─── Vout
          ┌──(−)──┤+         │
          │       └──────────┘
         [R1]
          │
         GND

   Gain: Av = 1 + Rf/R1 = 1 + {100_000}/{10_000} = {gain}

   Characteristics:
   - High input impedance (op-amp + input)
   - Output in phase with input
   - Gain ≥ 1 (unity gain follower when Rf=0, R1=∞)

   Expected output: {1.0} × {gain} = {1.0 * gain}V
""")
    print("   Netlist:")
    print(circuit.build_netlist())


def demo_inverting():
    """Demonstrate inverting amplifier."""
    print("\n2. Inverting Amplifier")
    print("-" * 40)

    circuit = Circuit("inverting_amp")

    vcc = Vdc("cc", 15.0)
    vee = Vdc("ee", -15.0)
    vin = Vdc("in", 1.0)

    rin = Resistor("in", resistance=10_000)
    rf = Resistor("f", resistance=100_000)
    opamp = create_component("opamp.lm741", "1")

    circuit.add(vcc, vee, vin, rin, rf, opamp)

    vcc_net = Net("vcc")
    vee_net = Net("vee")
    vin_net = Net("vin")
    vout = Net("vout")
    vminus = Net("vminus")

    # Power supplies
    circuit.connect(vcc.ports[0], vcc_net)
    circuit.connect(vcc.ports[1], GND)
    circuit.connect(vee.ports[0], GND)
    circuit.connect(vee.ports[1], vee_net)

    # Input
    circuit.connect(vin.ports[0], vin_net)
    circuit.connect(vin.ports[1], GND)
    circuit.connect(rin.ports[0], vin_net)
    circuit.connect(rin.ports[1], vminus)

    # Op-amp
    circuit.connect(opamp.ports[0], vout)
    circuit.connect(opamp.ports[1], GND)  # + input to ground
    circuit.connect(opamp.ports[2], vminus)  # - input
    circuit.connect(opamp.ports[3], vcc_net)
    circuit.connect(opamp.ports[4], vee_net)

    # Feedback
    circuit.connect(rf.ports[0], vout)
    circuit.connect(rf.ports[1], vminus)

    gain = -100_000 / 10_000

    print(f"""
   Inverting Amplifier

   Circuit:
               ┌───[Rf]───┐
               │          │
   Vin ─[Rin]──┤−         │
               │    741   ├─── Vout
        ┌──────┤+         │
        │      └──────────┘
       GND

   Gain: Av = -Rf/Rin = -{100_000}/{10_000} = {gain}

   Characteristics:
   - Input impedance ≈ Rin (virtual ground at - input)
   - Output is inverted (180° phase shift)
   - Virtual ground: V- ≈ 0V due to feedback

   Expected output: {1.0} × {gain} = {1.0 * gain}V
""")
    print("   Netlist:")
    print(circuit.build_netlist())


def demo_voltage_follower():
    """Demonstrate voltage follower (buffer)."""
    print("\n3. Voltage Follower (Buffer)")
    print("-" * 40)

    circuit = Circuit("voltage_follower")

    vcc = Vdc("cc", 15.0)
    vee = Vdc("ee", -15.0)
    vin = Vdc("in", 5.0)

    opamp = create_component("opamp.lm741", "1")

    circuit.add(vcc, vee, vin, opamp)

    vcc_net = Net("vcc")
    vee_net = Net("vee")
    vin_net = Net("vin")
    vout = Net("vout")

    # Power
    circuit.connect(vcc.ports[0], vcc_net)
    circuit.connect(vcc.ports[1], GND)
    circuit.connect(vee.ports[0], GND)
    circuit.connect(vee.ports[1], vee_net)

    # Input
    circuit.connect(vin.ports[0], vin_net)
    circuit.connect(vin.ports[1], GND)

    # Op-amp with unity feedback
    circuit.connect(opamp.ports[0], vout)
    circuit.connect(opamp.ports[1], vin_net)  # + input
    circuit.connect(opamp.ports[2], vout)  # - input tied to output
    circuit.connect(opamp.ports[3], vcc_net)
    circuit.connect(opamp.ports[4], vee_net)

    print("""
   Voltage Follower (Buffer)

   Circuit:
                  ┌──────────┐
                  │          │
   Vin ──(+)──────┤−    741  ├───┬── Vout
                  │+         │   │
          ┌───────┤          │   │
          │       └──────────┘   │
          └──────────────────────┘

   Gain: Av = 1 (unity)

   Characteristics:
   - Very high input impedance (10¹² Ω typical)
   - Very low output impedance (~75Ω)
   - No voltage gain, but huge current gain
   - Isolates source from load

   Applications:
   - Impedance matching
   - Buffer for high-impedance sensors
   - Driving capacitive loads
   - Signal isolation
""")
    print("   Netlist:")
    print(circuit.build_netlist())


def demo_summing_amplifier():
    """Demonstrate summing amplifier."""
    print("\n4. Summing Amplifier")
    print("-" * 40)

    circuit = Circuit("summing_amp")

    vcc = Vdc("cc", 15.0)
    vee = Vdc("ee", -15.0)
    v1 = Vdc("1", 1.0)
    v2 = Vdc("2", 2.0)
    v3 = Vdc("3", 0.5)

    r1 = Resistor("1", resistance=10_000)
    r2 = Resistor("2", resistance=10_000)
    r3 = Resistor("3", resistance=10_000)
    rf = Resistor("f", resistance=10_000)
    opamp = create_component("opamp.lm741", "1")

    circuit.add(vcc, vee, v1, v2, v3, r1, r2, r3, rf, opamp)

    vcc_net = Net("vcc")
    vee_net = Net("vee")
    v1_net = Net("v1")
    v2_net = Net("v2")
    v3_net = Net("v3")
    vout = Net("vout")
    vminus = Net("vminus")

    # Power
    circuit.connect(vcc.ports[0], vcc_net)
    circuit.connect(vcc.ports[1], GND)
    circuit.connect(vee.ports[0], GND)
    circuit.connect(vee.ports[1], vee_net)

    # Input voltages
    circuit.connect(v1.ports[0], v1_net)
    circuit.connect(v1.ports[1], GND)
    circuit.connect(v2.ports[0], v2_net)
    circuit.connect(v2.ports[1], GND)
    circuit.connect(v3.ports[0], v3_net)
    circuit.connect(v3.ports[1], GND)

    # Input resistors to summing node
    circuit.connect(r1.ports[0], v1_net)
    circuit.connect(r1.ports[1], vminus)
    circuit.connect(r2.ports[0], v2_net)
    circuit.connect(r2.ports[1], vminus)
    circuit.connect(r3.ports[0], v3_net)
    circuit.connect(r3.ports[1], vminus)

    # Op-amp
    circuit.connect(opamp.ports[0], vout)
    circuit.connect(opamp.ports[1], GND)
    circuit.connect(opamp.ports[2], vminus)
    circuit.connect(opamp.ports[3], vcc_net)
    circuit.connect(opamp.ports[4], vee_net)

    # Feedback
    circuit.connect(rf.ports[0], vout)
    circuit.connect(rf.ports[1], vminus)

    vout_expected = -(1.0 + 2.0 + 0.5)

    print(f"""
   Summing Amplifier

   Circuit:
   V1 ─[R1]─┐
            │  ┌───[Rf]───┐
   V2 ─[R2]─┼──┤−         │
            │  │    741   ├─── Vout
   V3 ─[R3]─┘  │+         │
        ┌──────┤          │
        │      └──────────┘
       GND

   Output: Vout = -Rf × (V1/R1 + V2/R2 + V3/R3)

   When R1 = R2 = R3 = Rf:
   Vout = -(V1 + V2 + V3) = -({1.0} + {2.0} + {0.5}) = {vout_expected}V

   Applications:
   - Audio mixing
   - Digital-to-analog conversion
   - Weighted summing (different R values)
   - Signal combining
""")
    print("   Netlist:")
    print(circuit.build_netlist())


def main():
    """Demonstrate op-amp circuits."""
    print("=" * 60)
    print("Tutorial 08: Operational Amplifiers")
    print("=" * 60)

    demo_non_inverting()
    demo_inverting()
    demo_voltage_follower()
    demo_summing_amplifier()

    print("\n" + "=" * 60)
    print("Summary: Op-Amp Configurations")
    print("=" * 60)
    print("""
   | Configuration   | Gain Formula    | Input Z   | Use Case        |
   |-----------------|-----------------|-----------|-----------------|
   | Non-inverting   | 1 + Rf/R1       | Very high | General amp     |
   | Inverting       | -Rf/Rin         | = Rin     | Precision amp   |
   | Follower        | 1               | Very high | Buffer          |
   | Summing         | -Rf × Σ(V/R)    | = Rin     | Signal mixing   |

   Common op-amps in library:
   - LM741: Classic, general purpose
   - LM358: Dual, single-supply capable
   - TL072: Low noise, JFET input
   - LM324: Quad, single-supply
""")
    # Show available op-amps
    spec = get_component_spec("opamp.lm741")
    print(f"\n   LM741: {spec.metadata.get('description', 'N/A')}")


if __name__ == "__main__":
    main()
