"""Linear Voltage Regulator

A basic series-pass linear regulator using an op-amp and transistor.

Run: python examples/power/linear_regulator.py
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def build_linear_regulator(v_out: float = 5.0) -> Circuit:
    """Build a linear voltage regulator.

    Args:
        v_out: Desired output voltage

    Returns:
        Circuit with the regulator
    """
    circuit = Circuit("linear_regulator")

    # Input supply (unregulated)
    vin = Vdc("in", 12.0)

    # Reference voltage (Zener-based)
    v_ref = 2.5  # Reference point
    r_zener = Resistor("zener", resistance=1000)
    d_zener = create_component("diode.1n4728a", "z")  # 3.3V Zener

    # Error amplifier
    opamp = create_component("opamp.lm741", "1")

    # Pass transistor (NPN Darlington for high current gain)
    q_pass = create_component("bjt.tip120", "pass")

    # Feedback divider
    # Vout = Vref × (1 + R1/R2)
    # R1/R2 = (Vout/Vref) - 1
    R2 = 10_000
    R1 = R2 * (v_out / v_ref - 1)

    r1 = Resistor("1", resistance=R1)
    r2 = Resistor("2", resistance=R2)

    # Output filter capacitor
    c_out = Capacitor("out", capacitance=100e-6)

    # Load resistor (example)
    r_load = Resistor("load", resistance=100)

    circuit.add(vin, r_zener, d_zener, opamp, q_pass, r1, r2, c_out, r_load)

    # Nets
    v_unreg = Net("v_unreg")
    v_ref_net = Net("v_ref")
    v_fb = Net("v_fb")
    v_drive = Net("v_drive")
    vout = Net("vout")

    # Input supply
    circuit.connect(vin.ports[0], v_unreg)
    circuit.connect(vin.ports[1], GND)

    # Zener reference
    circuit.connect(r_zener.ports[0], v_unreg)
    circuit.connect(r_zener.ports[1], v_ref_net)
    circuit.connect(d_zener.ports[1], v_ref_net)  # Cathode
    circuit.connect(d_zener.ports[0], GND)  # Anode

    # Op-amp error amplifier (out, +, -, V+, V-)
    circuit.connect(opamp.ports[0], v_drive)
    circuit.connect(opamp.ports[1], v_ref_net)  # + to reference
    circuit.connect(opamp.ports[2], v_fb)  # - to feedback
    circuit.connect(opamp.ports[3], v_unreg)  # V+ to unregulated
    circuit.connect(opamp.ports[4], GND)  # V- to ground

    # Pass transistor (C, B, E for NPN)
    circuit.connect(q_pass.ports[0], v_unreg)  # Collector to Vin
    circuit.connect(q_pass.ports[1], v_drive)  # Base driven by op-amp
    circuit.connect(q_pass.ports[2], vout)  # Emitter is output

    # Feedback divider
    circuit.connect(r1.ports[0], vout)
    circuit.connect(r1.ports[1], v_fb)
    circuit.connect(r2.ports[0], v_fb)
    circuit.connect(r2.ports[1], GND)

    # Output capacitor
    circuit.connect(c_out.ports[0], vout)
    circuit.connect(c_out.ports[1], GND)

    # Load
    circuit.connect(r_load.ports[0], vout)
    circuit.connect(r_load.ports[1], GND)

    return circuit, R1, R2, v_ref


def main():
    """Demonstrate linear voltage regulator."""
    print("=" * 60)
    print("Linear Voltage Regulator")
    print("=" * 60)

    v_out = 5.0
    circuit, R1, R2, v_ref = build_linear_regulator(v_out)

    print(f"""
   Series-Pass Linear Regulator

   Circuit topology:

   Vin ────┬──[Rz]──┬───────────────────┐
   (12V)   │       [Dz]                  │
           │        │                   C (TIP120)
           │       GND     ┌───(+)OP──B
           │               │            E
           ├───────────────┤             │
           │               │             ├── Vout (5V)
          V+              (-)            │
           │               │            [R1]
          OP              [fb]           │
           │               │            [R2]
          V-               │             │
           │               └─────────────┤
          GND                           GND

   Design:
   - Reference voltage: Vref = {v_ref}V (from Zener)
   - Feedback ratio: Vout/Vref = {v_out/v_ref:.2f}
   - R1 = {R1:.0f} Ω, R2 = {R2:.0f} Ω

   Operation:
   - Op-amp compares Vref to feedback voltage
   - Adjusts pass transistor to maintain Vout
   - Vout = Vref × (1 + R1/R2) = {v_ref} × {1 + R1/R2:.2f} = {v_out}V

   Characteristics:
   - Low output ripple
   - Fast transient response
   - Dropout: Vin - Vout > ~2V required
   - Power dissipation: P = (Vin - Vout) × Iload

   At 50mA load:
   - Power dissipation: (12 - 5) × 0.05 = 0.35W
""")

    print("   Netlist:")
    print(circuit.build_netlist())

    result = circuit.validate()
    print(f"\n   Validation: {'VALID' if result.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
