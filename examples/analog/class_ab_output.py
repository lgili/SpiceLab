"""Class AB Output Stage

A complementary push-pull output stage with bias for reduced crossover distortion.

Run: python examples/analog/class_ab_output.py
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def build_class_ab_output() -> Circuit:
    """Build a Class AB push-pull output stage.

    Returns:
        Circuit with the output stage
    """
    circuit = Circuit("class_ab_output")

    # Power supplies
    vcc = Vdc("cc", 15.0)
    vee = Vdc("ee", -15.0)

    # Bias network (Vbe multiplier)
    # Creates ~1.4V drop for two Vbe's worth of bias
    r_bias1 = Resistor("bias1", resistance=1000)
    r_bias2 = Resistor("bias2", resistance=1000)
    q_bias = create_component("bjt.2n3904", "bias")

    # Output transistors (complementary pair)
    q_npn = create_component("bjt.2n3904", "npn")  # Top (NPN)
    q_pnp = create_component("bjt.2n3906", "pnp")  # Bottom (PNP)

    # Current limiting resistors
    r_e_npn = Resistor("e_npn", resistance=0.47)
    r_e_pnp = Resistor("e_pnp", resistance=0.47)

    # Load
    r_load = Resistor("load", resistance=8)  # 8 ohm speaker

    # Driver current source (simplified with resistor)
    r_driver = Resistor("driver", resistance=4700)

    circuit.add(
        vcc, vee, r_bias1, r_bias2, q_bias, q_npn, q_pnp, r_e_npn, r_e_pnp, r_load, r_driver
    )

    # Nets
    vcc_net = Net("vcc")
    vee_net = Net("vee")
    v_driver = Net("v_driver")
    v_bias_top = Net("v_bias_top")
    v_bias_bot = Net("v_bias_bot")
    v_out_npn = Net("v_out_npn")
    v_out_pnp = Net("v_out_pnp")
    vout = Net("vout")

    # Power supplies
    circuit.connect(vcc.ports[0], vcc_net)
    circuit.connect(vcc.ports[1], GND)
    circuit.connect(vee.ports[0], GND)
    circuit.connect(vee.ports[1], vee_net)

    # Driver resistor from Vcc
    circuit.connect(r_driver.ports[0], vcc_net)
    circuit.connect(r_driver.ports[1], v_driver)

    # Vbe multiplier (bias generator)
    # R_bias1 from driver to bias transistor collector
    circuit.connect(r_bias1.ports[0], v_driver)
    circuit.connect(r_bias1.ports[1], v_bias_top)

    # Bias transistor (C, B, E)
    circuit.connect(q_bias.ports[0], v_bias_top)
    circuit.connect(q_bias.ports[1], v_bias_top)  # Collector-base tied
    circuit.connect(q_bias.ports[2], v_bias_bot)

    # R_bias2 from emitter to bottom
    circuit.connect(r_bias2.ports[0], v_bias_bot)
    circuit.connect(r_bias2.ports[1], vee_net)

    # NPN output transistor (top)
    circuit.connect(q_npn.ports[0], vcc_net)  # Collector to Vcc
    circuit.connect(q_npn.ports[1], v_bias_top)  # Base
    circuit.connect(q_npn.ports[2], v_out_npn)  # Emitter

    # NPN emitter resistor
    circuit.connect(r_e_npn.ports[0], v_out_npn)
    circuit.connect(r_e_npn.ports[1], vout)

    # PNP output transistor (bottom)
    circuit.connect(q_pnp.ports[2], vee_net)  # Emitter to Vee
    circuit.connect(q_pnp.ports[1], v_bias_bot)  # Base
    circuit.connect(q_pnp.ports[0], v_out_pnp)  # Collector

    # PNP emitter resistor
    circuit.connect(r_e_pnp.ports[0], v_out_pnp)
    circuit.connect(r_e_pnp.ports[1], vout)

    # Load
    circuit.connect(r_load.ports[0], vout)
    circuit.connect(r_load.ports[1], GND)

    return circuit


def main():
    """Demonstrate Class AB output stage."""
    print("=" * 60)
    print("Class AB Push-Pull Output Stage")
    print("=" * 60)

    circuit = build_class_ab_output()

    print("""
   Class AB Output Stage

   Circuit topology:

        Vcc (+15V)
         │
         ├──[Rdriver]
         │
         ├─┬──────── Q_npn (NPN)
         │ │              │
         │ Qbias          [Re]
         │ │              │
         ├─┴──────── Q_pnp (PNP)──┬── Vout
         │              │         │
         │             [Re]     [Rload]
         │              │         │
        Vee           Vee        GND

   Components:
   - NPN output: 2N3904 (positive half)
   - PNP output: 2N3906 (negative half)
   - Vbe multiplier: Creates 2×Vbe bias
   - Emitter resistors: 0.47Ω for stability

   Operation:
   - Positive input: NPN conducts, sources current
   - Negative input: PNP conducts, sinks current
   - Bias keeps both slightly ON at zero crossing
   - Eliminates crossover distortion

   Class comparison:
   - Class A: Both always ON (inefficient, no distortion)
   - Class B: Each ON for half cycle (crossover distortion)
   - Class AB: Small bias, best compromise

   Maximum output:
   - Vout_max ≈ ±(Vcc - Vce_sat - Ve) ≈ ±13V
   - Into 8Ω: P_max ≈ V²/(2R) ≈ 10.5W
""")

    print("   Netlist:")
    print(circuit.build_netlist())

    result = circuit.validate()
    print(f"\n   Validation: {'VALID' if result.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
