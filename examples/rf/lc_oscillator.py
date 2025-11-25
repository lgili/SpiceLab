"""LC Oscillators (Colpitts and Hartley)

LC oscillators generate sinusoidal signals using an LC tank circuit
for frequency determination. The Colpitts uses capacitive feedback,
while the Hartley uses inductive feedback.

Run: python examples/rf/lc_oscillator.py
"""

import math

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Inductor, Resistor
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def build_colpitts_oscillator(f_osc: float = 1e6) -> Circuit:
    """Build a Colpitts oscillator.

    The Colpitts oscillator uses a capacitive voltage divider
    (C1, C2) for feedback. Frequency is determined by L and
    the series combination of C1 and C2.

    Args:
        f_osc: Oscillation frequency in Hz

    Returns:
        Circuit with Colpitts oscillator
    """
    circuit = Circuit("colpitts_oscillator")

    # Design equations:
    # f = 1 / (2π × √(L × Ceq))
    # Ceq = C1 × C2 / (C1 + C2)
    # Feedback ratio: β = C1 / C2
    # For oscillation: gain × β > 1

    # Choose reasonable component values
    L = 10e-6  # 10 µH
    # Ceq = 1 / (4π²f²L)
    Ceq = 1 / (4 * math.pi**2 * f_osc**2 * L)
    # Let C1 = C2 for equal feedback
    C1 = 2 * Ceq
    C2 = 2 * Ceq

    # Transistor
    q1 = create_component("bjt.2n3904", "1")

    # Tank components
    l1 = Inductor("1", inductance=L)
    c1 = Capacitor("1", capacitance=C1)
    c2 = Capacitor("2", capacitance=C2)

    # Bias resistors
    r_b1 = Resistor("b1", resistance=47_000)
    r_b2 = Resistor("b2", resistance=10_000)
    r_e = Resistor("e", resistance=1000)
    r_c = Resistor("c", resistance=2200)

    # Bypass/coupling capacitors
    c_e = Capacitor("e", capacitance=100e-9)  # Emitter bypass
    c_out = Capacitor("out", capacitance=100e-12)  # Output coupling

    # Power supply
    v_cc = create_component("source.vdc", "cc")

    circuit.add(q1, l1, c1, c2, r_b1, r_b2, r_e, r_c, c_e, c_out, v_cc)

    # Nets
    vcc = Net("vcc")
    v_coll = Net("v_coll")
    v_base = Net("v_base")
    v_emit = Net("v_emit")
    v_tank = Net("v_tank")
    vout = Net("vout")

    # Power supply
    circuit.connect(v_cc.ports[0], vcc)
    circuit.connect(v_cc.ports[1], GND)

    # Bias network
    circuit.connect(r_b1.ports[0], vcc)
    circuit.connect(r_b1.ports[1], v_base)
    circuit.connect(r_b2.ports[0], v_base)
    circuit.connect(r_b2.ports[1], GND)

    # Transistor
    circuit.connect(q1.ports[0], v_coll)  # Collector
    circuit.connect(q1.ports[1], v_base)  # Base
    circuit.connect(q1.ports[2], v_emit)  # Emitter

    # Emitter resistor and bypass
    circuit.connect(r_e.ports[0], v_emit)
    circuit.connect(r_e.ports[1], GND)
    circuit.connect(c_e.ports[0], v_emit)
    circuit.connect(c_e.ports[1], GND)

    # Collector resistor (RFC in some designs)
    circuit.connect(r_c.ports[0], vcc)
    circuit.connect(r_c.ports[1], v_coll)

    # Tank circuit: L in parallel with series C1-C2
    circuit.connect(l1.ports[0], v_coll)
    circuit.connect(l1.ports[1], v_tank)

    # Capacitive divider (C1 from collector, C2 to emitter feedback)
    circuit.connect(c1.ports[0], v_coll)
    circuit.connect(c1.ports[1], v_tank)
    circuit.connect(c2.ports[0], v_tank)
    circuit.connect(c2.ports[1], v_emit)  # Feedback to emitter

    # Output coupling
    circuit.connect(c_out.ports[0], v_coll)
    circuit.connect(c_out.ports[1], vout)

    return circuit, L, C1, C2


def build_hartley_oscillator(f_osc: float = 1e6) -> Circuit:
    """Build a Hartley oscillator.

    The Hartley oscillator uses a tapped inductor (or two inductors)
    for feedback. Frequency determined by total inductance and C.

    Args:
        f_osc: Oscillation frequency in Hz

    Returns:
        Circuit with Hartley oscillator
    """
    circuit = Circuit("hartley_oscillator")

    # Design equations:
    # f = 1 / (2π × √(Ltotal × C))
    # Ltotal = L1 + L2 + 2M (if coupled)
    # For uncoupled: Ltotal = L1 + L2
    # Feedback ratio: β = L1 / L2

    C = 100e-12  # 100 pF
    # Ltotal = 1 / (4π²f²C)
    Ltotal = 1 / (4 * math.pi**2 * f_osc**2 * C)
    L1 = Ltotal / 2  # Split equally
    L2 = Ltotal / 2

    # Transistor
    q1 = create_component("bjt.2n3904", "1")

    # Tank components
    l1 = Inductor("1", inductance=L1)
    l2 = Inductor("2", inductance=L2)
    c_tank = Capacitor("tank", capacitance=C)

    # Bias resistors
    r_b1 = Resistor("b1", resistance=47_000)
    r_b2 = Resistor("b2", resistance=10_000)
    r_e = Resistor("e", resistance=1000)

    # RF choke for collector
    l_rfc = Inductor("rfc", inductance=100e-6)

    # Bypass/coupling capacitors
    c_e = Capacitor("e", capacitance=100e-9)
    c_out = Capacitor("out", capacitance=100e-12)
    c_b = Capacitor("b", capacitance=10e-9)  # Base coupling

    # Power supply
    v_cc = create_component("source.vdc", "cc")

    circuit.add(q1, l1, l2, c_tank, r_b1, r_b2, r_e, l_rfc, c_e, c_out, c_b, v_cc)

    # Nets
    vcc = Net("vcc")
    v_coll = Net("v_coll")
    v_base = Net("v_base")
    v_emit = Net("v_emit")
    v_tap = Net("v_tap")  # Inductor tap point
    vout = Net("vout")

    # Power supply
    circuit.connect(v_cc.ports[0], vcc)
    circuit.connect(v_cc.ports[1], GND)

    # Bias network
    circuit.connect(r_b1.ports[0], vcc)
    circuit.connect(r_b1.ports[1], v_base)
    circuit.connect(r_b2.ports[0], v_base)
    circuit.connect(r_b2.ports[1], GND)

    # Transistor
    circuit.connect(q1.ports[0], v_coll)
    circuit.connect(q1.ports[1], v_base)
    circuit.connect(q1.ports[2], v_emit)

    # Emitter resistor and bypass
    circuit.connect(r_e.ports[0], v_emit)
    circuit.connect(r_e.ports[1], GND)
    circuit.connect(c_e.ports[0], v_emit)
    circuit.connect(c_e.ports[1], GND)

    # RF choke to collector
    circuit.connect(l_rfc.ports[0], vcc)
    circuit.connect(l_rfc.ports[1], v_coll)

    # Tank circuit: L1-L2 tapped inductor in parallel with C
    circuit.connect(l1.ports[0], v_coll)
    circuit.connect(l1.ports[1], v_tap)  # Tap point
    circuit.connect(l2.ports[0], v_tap)
    circuit.connect(l2.ports[1], GND)

    # Tank capacitor
    circuit.connect(c_tank.ports[0], v_coll)
    circuit.connect(c_tank.ports[1], GND)

    # Feedback from tap to base
    circuit.connect(c_b.ports[0], v_tap)
    circuit.connect(c_b.ports[1], v_base)

    # Output coupling
    circuit.connect(c_out.ports[0], v_coll)
    circuit.connect(c_out.ports[1], vout)

    return circuit, L1, L2, C


def build_clapp_oscillator(f_osc: float = 1e6) -> Circuit:
    """Build a Clapp oscillator.

    The Clapp is a variation of the Colpitts with an additional
    capacitor in series with the inductor for improved frequency
    stability.

    Args:
        f_osc: Oscillation frequency in Hz

    Returns:
        Circuit with Clapp oscillator
    """
    circuit = Circuit("clapp_oscillator")

    # In Clapp, C3 << C1, C2 so frequency mainly determined by L and C3
    # f ≈ 1 / (2π × √(L × C3))

    L = 10e-6  # 10 µH
    C3 = 1 / (4 * math.pi**2 * f_osc**2 * L)
    C1 = 10 * C3  # Much larger than C3
    C2 = 10 * C3

    # Transistor
    q1 = create_component("bjt.2n3904", "1")

    # Tank components
    l1 = Inductor("1", inductance=L)
    c1 = Capacitor("1", capacitance=C1)
    c2 = Capacitor("2", capacitance=C2)
    c3 = Capacitor("3", capacitance=C3)  # Series with L

    # Bias resistors
    r_b1 = Resistor("b1", resistance=47_000)
    r_b2 = Resistor("b2", resistance=10_000)
    r_e = Resistor("e", resistance=1000)
    r_c = Resistor("c", resistance=2200)

    # Bypass capacitors
    c_e = Capacitor("e", capacitance=100e-9)
    c_out = Capacitor("out", capacitance=100e-12)

    # Power supply
    v_cc = create_component("source.vdc", "cc")

    circuit.add(q1, l1, c1, c2, c3, r_b1, r_b2, r_e, r_c, c_e, c_out, v_cc)

    # Nets
    vcc = Net("vcc")
    v_coll = Net("v_coll")
    v_base = Net("v_base")
    v_emit = Net("v_emit")
    v_lc = Net("v_lc")  # Junction of L and C3
    v_tank = Net("v_tank")
    vout = Net("vout")

    # Power supply
    circuit.connect(v_cc.ports[0], vcc)
    circuit.connect(v_cc.ports[1], GND)

    # Bias network
    circuit.connect(r_b1.ports[0], vcc)
    circuit.connect(r_b1.ports[1], v_base)
    circuit.connect(r_b2.ports[0], v_base)
    circuit.connect(r_b2.ports[1], GND)

    # Transistor
    circuit.connect(q1.ports[0], v_coll)
    circuit.connect(q1.ports[1], v_base)
    circuit.connect(q1.ports[2], v_emit)

    # Emitter resistor and bypass
    circuit.connect(r_e.ports[0], v_emit)
    circuit.connect(r_e.ports[1], GND)
    circuit.connect(c_e.ports[0], v_emit)
    circuit.connect(c_e.ports[1], GND)

    # Collector resistor
    circuit.connect(r_c.ports[0], vcc)
    circuit.connect(r_c.ports[1], v_coll)

    # Tank: L in series with C3, parallel with C1
    circuit.connect(l1.ports[0], v_coll)
    circuit.connect(l1.ports[1], v_lc)
    circuit.connect(c3.ports[0], v_lc)
    circuit.connect(c3.ports[1], v_tank)

    # Capacitive divider
    circuit.connect(c1.ports[0], v_coll)
    circuit.connect(c1.ports[1], v_tank)
    circuit.connect(c2.ports[0], v_tank)
    circuit.connect(c2.ports[1], v_emit)

    # Output coupling
    circuit.connect(c_out.ports[0], v_coll)
    circuit.connect(c_out.ports[1], vout)

    return circuit, L, C1, C2, C3


def main():
    """Demonstrate LC oscillator circuits."""
    print("=" * 60)
    print("LC Oscillators (Colpitts and Hartley)")
    print("=" * 60)

    f_osc = 1e6  # 1 MHz

    circuit1, L1, C1_a, C2_a = build_colpitts_oscillator(f_osc)
    circuit2, L1_h, L2_h, C_h = build_hartley_oscillator(f_osc)
    circuit3, L_c, C1_c, C2_c, C3_c = build_clapp_oscillator(f_osc)

    print(f"""
   LC Oscillators - Sinusoidal RF Signal Generation

   1. Colpitts Oscillator:

                Vcc
                 │
                [Rc]
                 │
        ┌───────┼───────┐
       [L]     [Q]     [C_out]── Vout
        │       │       │
        └───┬───┘       │
            │           │
           [C1]         │
            │           │
           [C2]─────────┘
            │
           [Re]
            │
           GND

   Frequency: f = 1 / (2π × √(L × Ceq))
   Ceq = C1×C2 / (C1+C2) = {C1_a*C2_a/(C1_a+C2_a)*1e12:.1f} pF

   Components:
   - L = {L1*1e6:.1f} µH
   - C1 = {C1_a*1e12:.1f} pF
   - C2 = {C2_a*1e12:.1f} pF

   2. Hartley Oscillator:

                Vcc
                 │
               [RFC]
                 │
        ┌───────┼───────┐
       [C]     [Q]     [C_out]── Vout
        │       │       │
        │   ┌───┘       │
        │   │           │
       [L1]─┴─[L2]      │
        │       │       │
       GND    [Cb]──────┘
               │
              base

   Frequency: f = 1 / (2π × √(Ltot × C))
   Ltot = L1 + L2 = {(L1_h+L2_h)*1e6:.1f} µH

   Components:
   - L1 = {L1_h*1e6:.2f} µH
   - L2 = {L2_h*1e6:.2f} µH
   - C = {C_h*1e12:.1f} pF

   3. Clapp Oscillator:

   Similar to Colpitts but with C3 in series with L.
   Better frequency stability because f mainly determined
   by L and C3 (smallest capacitor dominates).

   Frequency: f ≈ 1 / (2π × √(L × C3))

   Components:
   - L = {L_c*1e6:.1f} µH
   - C3 = {C3_c*1e12:.1f} pF (frequency determining)
   - C1 = {C1_c*1e12:.1f} pF (feedback)
   - C2 = {C2_c*1e12:.1f} pF (feedback)

   Comparison:
   ┌──────────┬─────────────────┬───────────────────┐
   │ Type     │ Feedback        │ Advantages        │
   ├──────────┼─────────────────┼───────────────────┤
   │ Colpitts │ Capacitive      │ Simple, reliable  │
   │ Hartley  │ Inductive       │ Easy tuning       │
   │ Clapp    │ Capacitive+     │ Best stability    │
   └──────────┴─────────────────┴───────────────────┘

   Target frequency: {f_osc/1e6:.1f} MHz
""")

    print("   Colpitts Netlist:")
    print(circuit1.build_netlist())

    result1 = circuit1.validate()
    result2 = circuit2.validate()
    result3 = circuit3.validate()
    print(f"\n   Colpitts Validation: {'VALID' if result1.is_valid else 'INVALID'}")
    print(f"   Hartley Validation: {'VALID' if result2.is_valid else 'INVALID'}")
    print(f"   Clapp Validation: {'VALID' if result3.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
