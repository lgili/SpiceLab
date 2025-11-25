"""Precision Rectifier (Super Diode)

A precision rectifier eliminates the diode forward voltage drop
using op-amp feedback. Essential for accurate AC-to-DC conversion
of small signals.

Run: python examples/signal_processing/precision_rectifier.py
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vsin
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def build_halfwave_precision_rectifier(gain: float = 1.0) -> Circuit:
    """Build a precision half-wave rectifier.

    Uses an op-amp to eliminate diode voltage drop. The op-amp
    compensates for the diode drop by driving the diode harder.

    Args:
        gain: Rectifier gain (set by R2/R1)

    Returns:
        Circuit with precision half-wave rectifier
    """
    circuit = Circuit("precision_halfwave_rectifier")

    R1 = 10_000
    R2 = gain * R1

    # Input
    v_in = Vsin("in", "0 1.0 1000")

    # Op-amp
    u1 = create_component("opamp.ideal", "1")

    # Diodes
    d1 = create_component("diode.1n4148", "1")  # Forward path
    d2 = create_component("diode.1n4148", "2")  # Clamp path

    # Resistors
    r1 = Resistor("1", resistance=R1)  # Input
    r2 = Resistor("2", resistance=R2)  # Feedback

    circuit.add(v_in, u1, d1, d2, r1, r2)

    # Nets
    vin = Net("vin")
    vout = Net("vout")
    v_inv = Net("v_inv")
    v_opout = Net("v_opout")

    # Input
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)

    # R1: Input to inverting node
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], v_inv)

    # Op-amp
    circuit.connect(u1.ports[0], v_inv)  # Inverting input
    circuit.connect(u1.ports[1], GND)  # Non-inverting to GND
    circuit.connect(u1.ports[2], v_opout)  # Output

    # D1: Conducts on negative input (inverted output positive)
    circuit.connect(d1.ports[0], v_opout)
    circuit.connect(d1.ports[1], vout)

    # D2: Clamps when D1 is off (prevents saturation)
    circuit.connect(d2.ports[0], v_opout)
    circuit.connect(d2.ports[1], v_inv)

    # R2: Feedback through D1 to inverting node
    circuit.connect(r2.ports[0], vout)
    circuit.connect(r2.ports[1], v_inv)

    return circuit, R1, R2


def build_fullwave_precision_rectifier() -> Circuit:
    """Build a precision full-wave rectifier.

    Uses two op-amps: one for half-wave rectification, another
    for summing to create full-wave output. Provides accurate
    absolute value function.

    Returns:
        Circuit with precision full-wave rectifier
    """
    circuit = Circuit("precision_fullwave_rectifier")

    R = 10_000  # All resistors equal for unity gain

    # Input
    v_in = Vsin("in", "0 1.0 1000")

    # Op-amps
    u1 = create_component("opamp.ideal", "1")  # Half-wave stage
    u2 = create_component("opamp.ideal", "2")  # Summing stage

    # Diodes
    d1 = create_component("diode.1n4148", "1")
    d2 = create_component("diode.1n4148", "2")

    # Resistors
    r1 = Resistor("1", resistance=R)
    r2 = Resistor("2", resistance=R)
    r3 = Resistor("3", resistance=R)
    r4 = Resistor("4", resistance=R)
    r5 = Resistor("5", resistance=R)
    r6 = Resistor("6", resistance=2 * R)  # 2R for proper scaling

    circuit.add(v_in, u1, u2, d1, d2, r1, r2, r3, r4, r5, r6)

    # Nets
    vin = Net("vin")
    vout = Net("vout")
    v_inv1 = Net("v_inv1")
    v_inv2 = Net("v_inv2")
    v_half = Net("v_half")
    v_op1out = Net("v_op1out")

    # Input
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)

    # First stage: Half-wave rectifier
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], v_inv1)

    circuit.connect(u1.ports[0], v_inv1)
    circuit.connect(u1.ports[1], GND)
    circuit.connect(u1.ports[2], v_op1out)

    circuit.connect(d1.ports[0], v_op1out)
    circuit.connect(d1.ports[1], v_half)

    circuit.connect(d2.ports[0], v_op1out)
    circuit.connect(d2.ports[1], v_inv1)

    circuit.connect(r2.ports[0], v_half)
    circuit.connect(r2.ports[1], v_inv1)

    # Second stage: Summing amplifier
    # Sums: -Vin (through R3) and -2*V_half (through R6)
    circuit.connect(r3.ports[0], vin)
    circuit.connect(r3.ports[1], v_inv2)

    circuit.connect(r6.ports[0], v_half)
    circuit.connect(r6.ports[1], v_inv2)

    circuit.connect(u2.ports[0], v_inv2)
    circuit.connect(u2.ports[1], GND)
    circuit.connect(u2.ports[2], vout)

    circuit.connect(r4.ports[0], vout)
    circuit.connect(r4.ports[1], v_inv2)

    return circuit


def build_absolute_value_circuit() -> Circuit:
    """Build an absolute value circuit.

    Alternative topology using two op-amps that provides
    |Vin| output for both positive and negative inputs.

    Returns:
        Circuit with absolute value function
    """
    circuit = Circuit("absolute_value")

    R = 10_000

    # Input
    v_in = Vsin("in", "0 1.0 1000")

    # Op-amps
    u1 = create_component("opamp.ideal", "1")
    u2 = create_component("opamp.ideal", "2")

    # Diodes
    d1 = create_component("diode.1n4148", "1")
    d2 = create_component("diode.1n4148", "2")

    # Resistors (all equal for unity gain)
    r1 = Resistor("1", resistance=R)
    r2 = Resistor("2", resistance=R)
    r3 = Resistor("3", resistance=R)
    r4 = Resistor("4", resistance=R)
    r5 = Resistor("5", resistance=R)

    circuit.add(v_in, u1, u2, d1, d2, r1, r2, r3, r4, r5)

    # Nets
    vin = Net("vin")
    vout = Net("vout")
    v1 = Net("v1")
    v2 = Net("v2")
    v_op1 = Net("v_op1")

    # Input
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)

    # First op-amp: inverter with diode steering
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], v1)

    circuit.connect(u1.ports[0], v1)
    circuit.connect(u1.ports[1], GND)
    circuit.connect(u1.ports[2], v_op1)

    # Diodes steer current based on polarity
    circuit.connect(d1.ports[0], v_op1)
    circuit.connect(d1.ports[1], v2)

    circuit.connect(d2.ports[0], v1)
    circuit.connect(d2.ports[1], v_op1)

    circuit.connect(r2.ports[0], v2)
    circuit.connect(r2.ports[1], v1)

    # Second op-amp: summing/output stage
    circuit.connect(r3.ports[0], vin)
    circuit.connect(r3.ports[1], Net("v_sum"))

    circuit.connect(r4.ports[0], v2)
    circuit.connect(r4.ports[1], Net("v_sum"))

    circuit.connect(u2.ports[0], Net("v_sum"))
    circuit.connect(u2.ports[1], GND)
    circuit.connect(u2.ports[2], vout)

    circuit.connect(r5.ports[0], vout)
    circuit.connect(r5.ports[1], Net("v_sum"))

    return circuit


def main():
    """Demonstrate precision rectifier circuits."""
    print("=" * 60)
    print("Precision Rectifier (Super Diode)")
    print("=" * 60)

    circuit1, R1, R2 = build_halfwave_precision_rectifier()
    circuit2 = build_fullwave_precision_rectifier()
    circuit3 = build_absolute_value_circuit()

    print(f"""
   Precision Rectifier - Eliminates Diode Drop

   1. Half-Wave Precision Rectifier:

                    ┌──|>|──────┬── Vout
                    │   D1      │
   Vin ──[R1]──┬────┴──|>|──┐   │
               │       D2   │  [R2]
               │            │   │
              [─]──────────[U1] │
               │                │
              GND              ─┘

   Operation:
   - Vin positive: D2 conducts, D1 off, Vout = 0
   - Vin negative: D1 conducts, Vout = -Vin × R2/R1

   Gain = R2/R1 = {R2/R1:.1f}

   2. Full-Wave Precision Rectifier:

   Stage 1: Half-wave rectifier
   Stage 2: Summing amp combines Vin and rectified signal

   Vout = |Vin| (absolute value)

   3. Absolute Value Circuit:

   Alternative 2-opamp topology providing |Vin|.

   Key Advantages over Simple Diode:
   - No 0.6V dead zone
   - Linear down to microvolts
   - Accurate gain control
   - Temperature stable

   Applications:
   - AC voltmeters
   - Audio level detection
   - Power measurement
   - Signal conditioning
   - Modulation/demodulation
""")

    print("   Half-Wave Netlist:")
    print(circuit1.build_netlist())

    result1 = circuit1.validate()
    result2 = circuit2.validate()
    result3 = circuit3.validate()
    print(f"\n   Half-Wave Validation: {'VALID' if result1.is_valid else 'INVALID'}")
    print(f"   Full-Wave Validation: {'VALID' if result2.is_valid else 'INVALID'}")
    print(f"   Abs Value Validation: {'VALID' if result3.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
