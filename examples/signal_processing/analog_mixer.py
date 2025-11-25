"""Analog Mixer (Gilbert Cell)

Analog mixers multiply two signals together, essential for
frequency conversion in radio systems. The Gilbert cell is
the classic double-balanced mixer topology.

Run: python examples/signal_processing/analog_mixer.py
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vsin
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def build_passive_mixer() -> Circuit:
    """Build a simple passive diode ring mixer.

    Uses four diodes in a ring configuration. Simple but
    has conversion loss and requires high LO drive.

    Returns:
        Circuit with passive ring mixer
    """
    circuit = Circuit("diode_ring_mixer")

    # RF input (signal to be mixed)
    v_rf = Vsin("rf", "0 0.1 10700000")  # 10.7 MHz IF

    # LO input (local oscillator)
    v_lo = Vsin("lo", "0 1.0 10000000")  # 10 MHz LO

    # Four matched diodes
    d1 = create_component("diode.1n4148", "1")
    d2 = create_component("diode.1n4148", "2")
    d3 = create_component("diode.1n4148", "3")
    d4 = create_component("diode.1n4148", "4")

    # Transformers simulated with coupled inductors (simplified)
    # Using resistors for this demonstration
    r_rf1 = Resistor("rf1", resistance=50)
    r_rf2 = Resistor("rf2", resistance=50)
    r_lo1 = Resistor("lo1", resistance=50)
    r_lo2 = Resistor("lo2", resistance=50)
    r_if = Resistor("if", resistance=50)

    circuit.add(v_rf, v_lo, d1, d2, d3, d4)
    circuit.add(r_rf1, r_rf2, r_lo1, r_lo2, r_if)

    # Nets
    v_rf_in = Net("rf_in")
    v_lo_in = Net("lo_in")
    v_if_out = Net("if_out")
    v_ring_a = Net("ring_a")
    v_ring_b = Net("ring_b")
    v_rf_ct = Net("rf_ct")  # RF center tap
    v_lo_ct = Net("lo_ct")  # LO center tap

    # RF input
    circuit.connect(v_rf.ports[0], v_rf_in)
    circuit.connect(v_rf.ports[1], GND)

    # LO input
    circuit.connect(v_lo.ports[0], v_lo_in)
    circuit.connect(v_lo.ports[1], GND)

    # RF transformer simulation
    circuit.connect(r_rf1.ports[0], v_rf_in)
    circuit.connect(r_rf1.ports[1], v_rf_ct)
    circuit.connect(r_rf2.ports[0], v_rf_ct)
    circuit.connect(r_rf2.ports[1], GND)

    # Diode ring
    # D1: ring_a to ring_b (forward)
    circuit.connect(d1.ports[0], v_ring_a)
    circuit.connect(d1.ports[1], v_ring_b)

    # D2: ring_b to rf_ct (forward)
    circuit.connect(d2.ports[0], v_ring_b)
    circuit.connect(d2.ports[1], v_rf_ct)

    # D3: rf_ct to ring_a (forward)
    circuit.connect(d3.ports[0], v_rf_ct)
    circuit.connect(d3.ports[1], v_ring_a)

    # D4: lo_ct to ring_b (for LO injection)
    circuit.connect(d4.ports[0], v_lo_ct)
    circuit.connect(d4.ports[1], v_ring_b)

    # LO transformer simulation
    circuit.connect(r_lo1.ports[0], v_lo_in)
    circuit.connect(r_lo1.ports[1], v_ring_a)
    circuit.connect(r_lo2.ports[0], v_ring_b)
    circuit.connect(r_lo2.ports[1], v_lo_ct)

    # IF output
    circuit.connect(r_if.ports[0], v_ring_a)
    circuit.connect(r_if.ports[1], v_if_out)

    return circuit


def build_gilbert_cell_mixer() -> Circuit:
    """Build a Gilbert cell (active double-balanced mixer).

    The classic IC mixer topology providing gain, good isolation,
    and low spurious outputs.

    Returns:
        Circuit with Gilbert cell mixer
    """
    circuit = Circuit("gilbert_cell_mixer")

    # RF input (differential)
    v_rf_p = Vsin("rfp", "0 0.01 10000000")  # 10 MHz RF
    v_rf_n = Vsin("rfn", "0 -0.01 10000000")

    # LO input (differential, larger amplitude)
    v_lo_p = Vsin("lop", "0 0.3 9000000")  # 9 MHz LO
    v_lo_n = Vsin("lon", "0 -0.3 9000000")

    # Bias voltage
    v_bias = create_component("source.vdc", "bias")

    # Six transistors
    # Bottom pair (RF input)
    q1 = create_component("bjt.2n3904", "1")
    q2 = create_component("bjt.2n3904", "2")

    # Top quad (LO switching)
    q3 = create_component("bjt.2n3904", "3")
    q4 = create_component("bjt.2n3904", "4")
    q5 = create_component("bjt.2n3904", "5")
    q6 = create_component("bjt.2n3904", "6")

    # Tail current source (resistor for simplicity)
    r_tail = Resistor("tail", resistance=1000)

    # Collector load resistors
    r_c1 = Resistor("c1", resistance=1000)
    r_c2 = Resistor("c2", resistance=1000)

    # Emitter degeneration for linearity
    r_e1 = Resistor("e1", resistance=100)
    r_e2 = Resistor("e2", resistance=100)

    # Power supply
    v_cc = create_component("source.vdc", "cc")

    circuit.add(v_rf_p, v_rf_n, v_lo_p, v_lo_n, v_bias, v_cc)
    circuit.add(q1, q2, q3, q4, q5, q6)
    circuit.add(r_tail, r_c1, r_c2, r_e1, r_e2)

    # Nets
    vcc = Net("vcc")
    v_tail = Net("v_tail")
    v_e1 = Net("v_e1")
    v_e2 = Net("v_e2")
    v_c1 = Net("v_c1")  # Q1 collector
    v_c2 = Net("v_c2")  # Q2 collector
    v_if_p = Net("if_p")  # IF output +
    v_if_n = Net("if_n")  # IF output -
    v_rf_p_net = Net("rf_p")
    v_rf_n_net = Net("rf_n")
    v_lo_p_net = Net("lo_p")
    v_lo_n_net = Net("lo_n")
    v_bias_net = Net("bias")

    # Power supply
    circuit.connect(v_cc.ports[0], vcc)
    circuit.connect(v_cc.ports[1], GND)

    # RF inputs
    circuit.connect(v_rf_p.ports[0], v_rf_p_net)
    circuit.connect(v_rf_p.ports[1], GND)
    circuit.connect(v_rf_n.ports[0], v_rf_n_net)
    circuit.connect(v_rf_n.ports[1], GND)

    # LO inputs
    circuit.connect(v_lo_p.ports[0], v_lo_p_net)
    circuit.connect(v_lo_p.ports[1], GND)
    circuit.connect(v_lo_n.ports[0], v_lo_n_net)
    circuit.connect(v_lo_n.ports[1], GND)

    # Bias
    circuit.connect(v_bias.ports[0], v_bias_net)
    circuit.connect(v_bias.ports[1], GND)

    # Bottom differential pair (RF input)
    circuit.connect(q1.ports[0], v_c1)  # Collector
    circuit.connect(q1.ports[1], v_rf_p_net)  # Base
    circuit.connect(q1.ports[2], v_e1)  # Emitter

    circuit.connect(q2.ports[0], v_c2)  # Collector
    circuit.connect(q2.ports[1], v_rf_n_net)  # Base
    circuit.connect(q2.ports[2], v_e2)  # Emitter

    # Emitter resistors
    circuit.connect(r_e1.ports[0], v_e1)
    circuit.connect(r_e1.ports[1], v_tail)
    circuit.connect(r_e2.ports[0], v_e2)
    circuit.connect(r_e2.ports[1], v_tail)

    # Tail current source
    circuit.connect(r_tail.ports[0], v_tail)
    circuit.connect(r_tail.ports[1], GND)

    # Top switching quad
    # Q3, Q4 driven by LO, collectors cross-coupled
    circuit.connect(q3.ports[0], v_if_p)  # Collector to IF+
    circuit.connect(q3.ports[1], v_lo_p_net)  # Base to LO+
    circuit.connect(q3.ports[2], v_c1)  # Emitter to Q1 collector

    circuit.connect(q4.ports[0], v_if_n)  # Collector to IF-
    circuit.connect(q4.ports[1], v_lo_n_net)  # Base to LO-
    circuit.connect(q4.ports[2], v_c1)  # Emitter to Q1 collector

    # Q5, Q6 (other half)
    circuit.connect(q5.ports[0], v_if_n)  # Collector to IF-
    circuit.connect(q5.ports[1], v_lo_p_net)  # Base to LO+
    circuit.connect(q5.ports[2], v_c2)  # Emitter to Q2 collector

    circuit.connect(q6.ports[0], v_if_p)  # Collector to IF+
    circuit.connect(q6.ports[1], v_lo_n_net)  # Base to LO-
    circuit.connect(q6.ports[2], v_c2)  # Emitter to Q2 collector

    # Collector load resistors
    circuit.connect(r_c1.ports[0], vcc)
    circuit.connect(r_c1.ports[1], v_if_p)
    circuit.connect(r_c2.ports[0], vcc)
    circuit.connect(r_c2.ports[1], v_if_n)

    return circuit


def build_simple_modulator() -> Circuit:
    """Build a simple AM modulator using transistor.

    Demonstrates basic signal multiplication for
    amplitude modulation.

    Returns:
        Circuit with simple AM modulator
    """
    circuit = Circuit("am_modulator")

    # Carrier signal
    v_carrier = Vsin("car", "0 1.0 1000000")  # 1 MHz carrier

    # Modulating signal (audio)
    v_mod = Vsin("mod", "2.5 0.5 1000")  # 1 kHz modulation + DC bias

    # Transistor as multiplier
    q1 = create_component("bjt.2n3904", "1")

    # Bias and coupling
    r_b = Resistor("b", resistance=100_000)
    r_c = Resistor("c", resistance=1000)
    r_e = Resistor("e", resistance=100)

    c_in = Capacitor("in", capacitance=100e-9)
    c_out = Capacitor("out", capacitance=100e-9)

    # Power supply
    v_cc = create_component("source.vdc", "cc")

    circuit.add(v_carrier, v_mod, v_cc, q1, r_b, r_c, r_e, c_in, c_out)

    # Nets
    vcc = Net("vcc")
    v_car_in = Net("car_in")
    v_mod_in = Net("mod_in")
    v_base = Net("base")
    v_coll = Net("coll")
    vout = Net("vout")

    # Power
    circuit.connect(v_cc.ports[0], vcc)
    circuit.connect(v_cc.ports[1], GND)

    # Carrier input
    circuit.connect(v_carrier.ports[0], v_car_in)
    circuit.connect(v_carrier.ports[1], GND)

    # Modulation input
    circuit.connect(v_mod.ports[0], v_mod_in)
    circuit.connect(v_mod.ports[1], GND)

    # Carrier coupled to base
    circuit.connect(c_in.ports[0], v_car_in)
    circuit.connect(c_in.ports[1], v_base)

    # Base bias from modulation (varies bias point)
    circuit.connect(r_b.ports[0], v_mod_in)
    circuit.connect(r_b.ports[1], v_base)

    # Transistor
    circuit.connect(q1.ports[0], v_coll)  # Collector
    circuit.connect(q1.ports[1], v_base)  # Base
    circuit.connect(q1.ports[2], Net("emitter"))  # Emitter

    # Emitter resistor
    circuit.connect(r_e.ports[0], Net("emitter"))
    circuit.connect(r_e.ports[1], GND)

    # Collector resistor
    circuit.connect(r_c.ports[0], vcc)
    circuit.connect(r_c.ports[1], v_coll)

    # Output coupling
    circuit.connect(c_out.ports[0], v_coll)
    circuit.connect(c_out.ports[1], vout)

    return circuit


def main():
    """Demonstrate analog mixer circuits."""
    print("=" * 60)
    print("Analog Mixer (Gilbert Cell)")
    print("=" * 60)

    circuit1 = build_passive_mixer()
    circuit2 = build_gilbert_cell_mixer()
    circuit3 = build_simple_modulator()

    print("""
   Analog Mixer - Frequency Conversion

   Mixers perform multiplication: Vout = Vrf × Vlo
   This creates sum and difference frequencies:
   fout = |frf ± flo|

   1. Passive Diode Ring Mixer:

         ┌──D1──┐
   RF ───┤      ├─── IF
         └──D2──┘
            │
           LO

   - Four diodes in ring/bridge configuration
   - Conversion loss: ~6 dB
   - Requires strong LO drive (~+7 dBm)
   - Good port isolation
   - Wide bandwidth

   2. Gilbert Cell (Active Mixer):

              Vcc
               │
          ┌────┼────┐
         [Rc1]   [Rc2]
          │       │
     ┌──Q3─┬─Q4──┬─Q5─┬─Q6──┐
     │  LO+│ LO- │LO+ │ LO- │
     │     └──┬──┘    └──┬──┘
     │        │          │
     │       Q1         Q2
     │      RF+         RF-
     │        └────┬────┘
     │            [Re]
     IF+          [Itail]
                  GND

   Advantages:
   - Conversion gain (not loss)
   - On-chip integration
   - Lower LO drive required
   - Excellent isolation

   Key specs:
   - Conversion gain/loss
   - Noise figure
   - IP3 (linearity)
   - LO-RF isolation
   - Port VSWR

   3. AM Modulator:

   Carrier ──||──┬──[Q1]── Output
                 │    │
   Audio ──[Rb]──┘   [Rc]
                      │
                     Vcc

   - Transistor gain varies with audio
   - Output = carrier × (1 + m·audio)
   - Creates AM sidebands

   Applications:
   - Radio receivers (superheterodyne)
   - Transmitters (upconversion)
   - Phase-locked loops
   - Spectrum analyzers
   - Modulation/demodulation
""")

    print("   Gilbert Cell Netlist:")
    print(circuit2.build_netlist())

    result1 = circuit1.validate()
    result2 = circuit2.validate()
    result3 = circuit3.validate()
    print(f"\n   Passive Mixer Validation: {'VALID' if result1.is_valid else 'INVALID'}")
    print(f"   Gilbert Cell Validation: {'VALID' if result2.is_valid else 'INVALID'}")
    print(f"   AM Modulator Validation: {'VALID' if result3.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
