"""Schmitt Trigger

A hysteresis comparator using positive feedback.

Run: python examples/analog/schmitt_trigger.py
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def build_inverting_schmitt(
    v_ref: float = 0.0,
    vth_high: float = 3.0,
    vth_low: float = 2.0,
) -> Circuit:
    """Build an inverting Schmitt trigger.

    Args:
        v_ref: Reference voltage (center point)
        vth_high: Upper threshold voltage
        vth_low: Lower threshold voltage

    Returns:
        Circuit with the Schmitt trigger
    """
    circuit = Circuit("schmitt_trigger")

    # Power supplies
    vcc = Vdc("cc", 15.0)
    vee = Vdc("ee", -15.0)

    # Calculate resistors for desired thresholds
    # For inverting configuration with ±Vsat:
    # Vth+ = Vref + (Vsat - Vref) × R1/(R1+R2)
    # Vth- = Vref + (-Vsat - Vref) × R1/(R1+R2)

    # Hysteresis = Vth+ - Vth- = 2 × Vsat × R1/(R1+R2)
    Vsat = 13.5  # Typical op-amp saturation voltage
    hysteresis = vth_high - vth_low

    # R1/(R1+R2) = hysteresis / (2 × Vsat)
    ratio = hysteresis / (2 * Vsat)

    R1 = 10_000
    R2 = R1 * (1 - ratio) / ratio

    r1 = Resistor("1", resistance=R1)
    r2 = Resistor("2", resistance=R2)

    # Input resistor
    rin = Resistor("in", resistance=10_000)

    opamp = create_component("opamp.lm741", "1")

    circuit.add(vcc, vee, r1, r2, rin, opamp)

    # Nets
    vcc_net = Net("vcc")
    vee_net = Net("vee")
    vin = Net("vin")
    vplus = Net("vplus")
    vout = Net("vout")

    # Power supplies
    circuit.connect(vcc.ports[0], vcc_net)
    circuit.connect(vcc.ports[1], GND)
    circuit.connect(vee.ports[0], GND)
    circuit.connect(vee.ports[1], vee_net)

    # Input through resistor to inverting input
    circuit.connect(rin.ports[0], vin)
    circuit.connect(rin.ports[1], Net("vminus"))

    # Positive feedback network
    circuit.connect(r1.ports[0], vout)
    circuit.connect(r1.ports[1], vplus)
    circuit.connect(r2.ports[0], vplus)
    circuit.connect(r2.ports[1], GND)  # Vref = 0V

    # Op-amp
    circuit.connect(opamp.ports[0], vout)
    circuit.connect(opamp.ports[1], vplus)
    circuit.connect(opamp.ports[2], Net("vminus"))
    circuit.connect(opamp.ports[3], vcc_net)
    circuit.connect(opamp.ports[4], vee_net)

    return circuit, R1, R2


def build_noninverting_schmitt(hysteresis: float = 1.0) -> Circuit:
    """Build a non-inverting Schmitt trigger.

    Args:
        hysteresis: Total hysteresis voltage

    Returns:
        Circuit with the Schmitt trigger
    """
    circuit = Circuit("schmitt_noninv")

    # Power supplies
    vcc = Vdc("cc", 15.0)
    vee = Vdc("ee", -15.0)

    # For non-inverting:
    # Hysteresis = 2 × Vsat × R1/(R1+R2)
    Vsat = 13.5
    ratio = hysteresis / (2 * Vsat)

    R1 = 10_000
    R2 = R1 * (1 - ratio) / ratio

    r1 = Resistor("1", resistance=R1)
    r2 = Resistor("2", resistance=R2)

    opamp = create_component("opamp.lm741", "1")

    circuit.add(vcc, vee, r1, r2, opamp)

    # Nets
    vcc_net = Net("vcc")
    vee_net = Net("vee")
    vin = Net("vin")
    vplus = Net("vplus")
    vout = Net("vout")

    # Power supplies
    circuit.connect(vcc.ports[0], vcc_net)
    circuit.connect(vcc.ports[1], GND)
    circuit.connect(vee.ports[0], GND)
    circuit.connect(vee.ports[1], vee_net)

    # Positive feedback divider
    circuit.connect(r2.ports[0], vin)
    circuit.connect(r2.ports[1], vplus)
    circuit.connect(r1.ports[0], vplus)
    circuit.connect(r1.ports[1], vout)

    # Op-amp (inverting input grounded)
    circuit.connect(opamp.ports[0], vout)
    circuit.connect(opamp.ports[1], vplus)
    circuit.connect(opamp.ports[2], GND)
    circuit.connect(opamp.ports[3], vcc_net)
    circuit.connect(opamp.ports[4], vee_net)

    return circuit, R1, R2


def main():
    """Demonstrate Schmitt trigger circuits."""
    print("=" * 60)
    print("Schmitt Trigger (Hysteresis Comparator)")
    print("=" * 60)

    print("\n1. Inverting Schmitt Trigger")
    print("-" * 40)

    vth_high = 3.0
    vth_low = 2.0
    circuit1, R1_1, R2_1 = build_inverting_schmitt(vth_high=vth_high, vth_low=vth_low)

    print(f"""
   Inverting Schmitt Trigger

   Circuit:
                    R1
                ┌───┤├───┐
                │        │
   Vin ─[Rin]──(-)───────┤
                         │
           ┌───(+)──[R2]─┼── Vout
           │       │     │
           │      GND    │
           │             │
           └─────────────┘

   Design:
   - R1 = {R1_1:.0f} Ω
   - R2 = {R2_1:.0f} Ω
   - Upper threshold: Vth+ = {vth_high}V
   - Lower threshold: Vth- = {vth_low}V
   - Hysteresis: {vth_high - vth_low}V

   Operation:
   - Vin rising above Vth+ → Vout goes LOW
   - Vin falling below Vth- → Vout goes HIGH
   - Output inverted from input
""")
    print("   Netlist:")
    print(circuit1.build_netlist())

    print("\n2. Non-Inverting Schmitt Trigger")
    print("-" * 40)

    hysteresis = 1.0
    circuit2, R1_2, R2_2 = build_noninverting_schmitt(hysteresis=hysteresis)

    print(f"""
   Non-Inverting Schmitt Trigger

   Circuit:
              R2
   Vin ───────┤├───┬───(+)───┐
                   │         │
                  [R1]──────[OP]── Vout
                   │         │
                  Vout    (-)│
                             │
                            GND

   Design:
   - R1 = {R1_2:.0f} Ω
   - R2 = {R2_2:.0f} Ω
   - Hysteresis: ±{hysteresis/2}V around 0V

   Operation:
   - Input mixed with output via R1, R2 divider
   - Positive feedback creates snap action
   - Output follows input polarity

   Applications:
   - Noise-immune switching
   - Square wave generation from sine
   - Contact debouncing
   - Level detection with immunity
""")
    print("   Netlist:")
    print(circuit2.build_netlist())


if __name__ == "__main__":
    main()
