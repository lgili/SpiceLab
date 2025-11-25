"""Crystal Oscillator (Pierce Configuration)

Crystal oscillators provide extremely stable frequency references
using the mechanical resonance of a quartz crystal. The Pierce
configuration is the most common IC-compatible topology.

Run: python examples/rf/crystal_oscillator.py
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Inductor, Resistor
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def build_pierce_oscillator(f_crystal: float = 10e6) -> Circuit:
    """Build a Pierce crystal oscillator.

    The Pierce oscillator uses a crystal in parallel resonance
    with two capacitors (C1, C2) forming a pi network. The
    inverter provides gain.

    Args:
        f_crystal: Crystal frequency in Hz

    Returns:
        Circuit with Pierce oscillator
    """
    circuit = Circuit("pierce_oscillator")

    # Crystal model: Series RLC with parallel capacitance
    # Motional parameters (typical 10MHz crystal):
    Lm = 10e-3  # 10 mH motional inductance
    Cm = 25e-15  # 25 fF motional capacitance
    Rm = 10  # 10Ω motional resistance (ESR)
    C0 = 5e-12  # 5 pF parallel (holder) capacitance

    # Load capacitors (determines exact frequency)
    # Crystal pulls slightly with load capacitance
    C1 = 22e-12  # 22 pF
    C2 = 22e-12  # 22 pF

    # Crystal model components
    l_m = Inductor("m", inductance=Lm)
    c_m = Capacitor("m", capacitance=Cm)
    r_m = Resistor("m", resistance=Rm)
    c_0 = Capacitor("0", capacitance=C0)

    # Load capacitors
    c_1 = Capacitor("1", capacitance=C1)
    c_2 = Capacitor("2", capacitance=C2)

    # Feedback resistor (provides DC bias for inverter)
    r_f = Resistor("f", resistance=1e6)  # 1MΩ

    # Current limiting resistor (optional, for start-up)
    r_s = Resistor("s", resistance=1000)

    # Inverter using transistor (could use CMOS inverter in IC)
    q1 = create_component("bjt.2n3904", "1")

    # Bias resistors
    r_b1 = Resistor("b1", resistance=47_000)
    r_b2 = Resistor("b2", resistance=47_000)
    r_e = Resistor("e", resistance=1000)
    r_c = Resistor("c", resistance=2200)

    # Bypass
    c_e = Capacitor("e", capacitance=100e-9)

    # Power supply
    v_cc = create_component("source.vdc", "cc")

    circuit.add(l_m, c_m, r_m, c_0, c_1, c_2, r_f, r_s)
    circuit.add(q1, r_b1, r_b2, r_e, r_c, c_e, v_cc)

    # Nets
    vcc = Net("vcc")
    v_in = Net("v_in")  # Crystal input (base side)
    v_out = Net("v_out")  # Crystal output (collector side)
    v_base = Net("v_base")
    v_coll = Net("v_coll")
    v_emit = Net("v_emit")
    v_xtal = Net("v_xtal")  # Between series RLC elements

    # Power supply
    circuit.connect(v_cc.ports[0], vcc)
    circuit.connect(v_cc.ports[1], GND)

    # Bias network
    circuit.connect(r_b1.ports[0], vcc)
    circuit.connect(r_b1.ports[1], v_base)
    circuit.connect(r_b2.ports[0], v_base)
    circuit.connect(r_b2.ports[1], GND)

    # Transistor (inverting amplifier)
    circuit.connect(q1.ports[0], v_coll)
    circuit.connect(q1.ports[1], v_base)
    circuit.connect(q1.ports[2], v_emit)

    # Emitter resistor with bypass
    circuit.connect(r_e.ports[0], v_emit)
    circuit.connect(r_e.ports[1], GND)
    circuit.connect(c_e.ports[0], v_emit)
    circuit.connect(c_e.ports[1], GND)

    # Collector resistor
    circuit.connect(r_c.ports[0], vcc)
    circuit.connect(r_c.ports[1], v_coll)

    # Crystal: Parallel C0 with series RLC
    # Series RLC path
    circuit.connect(r_m.ports[0], v_in)
    circuit.connect(r_m.ports[1], v_xtal)
    circuit.connect(l_m.ports[0], v_xtal)
    circuit.connect(l_m.ports[1], Net("v_xtal2"))
    circuit.connect(c_m.ports[0], Net("v_xtal2"))
    circuit.connect(c_m.ports[1], v_out)

    # Parallel capacitance
    circuit.connect(c_0.ports[0], v_in)
    circuit.connect(c_0.ports[1], v_out)

    # Load capacitors
    circuit.connect(c_1.ports[0], v_in)
    circuit.connect(c_1.ports[1], GND)
    circuit.connect(c_2.ports[0], v_out)
    circuit.connect(c_2.ports[1], GND)

    # Feedback resistor (DC bias)
    circuit.connect(r_f.ports[0], v_in)
    circuit.connect(r_f.ports[1], v_out)

    # Connect to inverter
    circuit.connect(r_s.ports[0], v_base)
    circuit.connect(r_s.ports[1], v_in)

    # Output from collector (inverted) feeds back through crystal
    circuit.connect(v_out, v_coll)

    return circuit, C1, C2, Lm, Cm, Rm, C0


def build_colpitts_crystal_oscillator(f_crystal: float = 10e6) -> Circuit:
    """Build a Colpitts crystal oscillator.

    Uses the crystal in place of the inductor in a Colpitts
    topology. The crystal operates in series resonance.

    Args:
        f_crystal: Crystal frequency in Hz

    Returns:
        Circuit with Colpitts crystal oscillator
    """
    circuit = Circuit("colpitts_crystal_oscillator")

    # Crystal model (series mode)
    Lm = 10e-3
    Cm = 25e-15
    Rm = 10
    C0 = 5e-12

    # Colpitts capacitors
    C1 = 100e-12
    C2 = 100e-12

    # Crystal components
    l_m = Inductor("m", inductance=Lm)
    c_m = Capacitor("m", capacitance=Cm)
    r_m = Resistor("m", resistance=Rm)
    c_0 = Capacitor("0", capacitance=C0)

    # Colpitts caps
    c_1 = Capacitor("1", capacitance=C1)
    c_2 = Capacitor("2", capacitance=C2)

    # Transistor
    q1 = create_component("bjt.2n3904", "1")

    # Bias
    r_b1 = Resistor("b1", resistance=47_000)
    r_b2 = Resistor("b2", resistance=10_000)
    r_e = Resistor("e", resistance=1000)

    # RFC
    l_rfc = Inductor("rfc", inductance=100e-6)

    # Bypass
    c_e = Capacitor("e", capacitance=100e-9)
    c_out = Capacitor("out", capacitance=100e-12)

    # Power supply
    v_cc = create_component("source.vdc", "cc")

    circuit.add(l_m, c_m, r_m, c_0, c_1, c_2)
    circuit.add(q1, r_b1, r_b2, r_e, l_rfc, c_e, c_out, v_cc)

    # Nets
    vcc = Net("vcc")
    v_base = Net("v_base")
    v_coll = Net("v_coll")
    v_emit = Net("v_emit")
    v_tank = Net("v_tank")
    v_xtal1 = Net("v_xtal1")
    v_xtal2 = Net("v_xtal2")
    vout = Net("vout")

    # Power
    circuit.connect(v_cc.ports[0], vcc)
    circuit.connect(v_cc.ports[1], GND)

    # Bias
    circuit.connect(r_b1.ports[0], vcc)
    circuit.connect(r_b1.ports[1], v_base)
    circuit.connect(r_b2.ports[0], v_base)
    circuit.connect(r_b2.ports[1], GND)

    # Transistor
    circuit.connect(q1.ports[0], v_coll)
    circuit.connect(q1.ports[1], v_base)
    circuit.connect(q1.ports[2], v_emit)

    # Emitter
    circuit.connect(r_e.ports[0], v_emit)
    circuit.connect(r_e.ports[1], GND)
    circuit.connect(c_e.ports[0], v_emit)
    circuit.connect(c_e.ports[1], GND)

    # RFC to collector
    circuit.connect(l_rfc.ports[0], vcc)
    circuit.connect(l_rfc.ports[1], v_coll)

    # Crystal from collector to tank
    circuit.connect(r_m.ports[0], v_coll)
    circuit.connect(r_m.ports[1], v_xtal1)
    circuit.connect(l_m.ports[0], v_xtal1)
    circuit.connect(l_m.ports[1], v_xtal2)
    circuit.connect(c_m.ports[0], v_xtal2)
    circuit.connect(c_m.ports[1], v_tank)

    # Parallel capacitance
    circuit.connect(c_0.ports[0], v_coll)
    circuit.connect(c_0.ports[1], v_tank)

    # Colpitts capacitors
    circuit.connect(c_1.ports[0], v_tank)
    circuit.connect(c_1.ports[1], GND)
    circuit.connect(c_2.ports[0], v_tank)
    circuit.connect(c_2.ports[1], v_emit)

    # Output
    circuit.connect(c_out.ports[0], v_coll)
    circuit.connect(c_out.ports[1], vout)

    return circuit


def build_overtone_crystal_oscillator(f_fundamental: float = 10e6) -> Circuit:
    """Build an overtone crystal oscillator.

    Crystals can oscillate at odd harmonics (3rd, 5th, 7th...).
    An LC tank selects the desired overtone.

    Args:
        f_fundamental: Fundamental crystal frequency

    Returns:
        Circuit oscillating at 3rd overtone
    """
    circuit = Circuit("overtone_crystal_oscillator")

    f_overtone = 3 * f_fundamental  # 3rd overtone

    # Crystal (fundamental mode parameters)
    Lm = 10e-3
    Cm = 25e-15
    Rm = 10
    C0 = 5e-12

    # Overtone selector tank
    # Tuned to reject fundamental, pass 3rd overtone
    L_tank = 1e-6
    C_tank = 1 / (4 * 3.14159**2 * f_overtone**2 * L_tank)

    # Crystal components
    l_m = Inductor("m", inductance=Lm)
    c_m = Capacitor("m", capacitance=Cm)
    r_m = Resistor("m", resistance=Rm)
    c_0 = Capacitor("0", capacitance=C0)

    # Tank circuit
    l_tank = Inductor("tank", inductance=L_tank)
    c_tank_comp = Capacitor("tank", capacitance=C_tank)

    # Load capacitors
    c_1 = Capacitor("1", capacitance=22e-12)
    c_2 = Capacitor("2", capacitance=22e-12)

    # Transistor
    q1 = create_component("bjt.2n3904", "1")

    # Bias
    r_b1 = Resistor("b1", resistance=47_000)
    r_b2 = Resistor("b2", resistance=47_000)
    r_e = Resistor("e", resistance=1000)
    r_c = Resistor("c", resistance=1000)

    # Bypass
    c_e = Capacitor("e", capacitance=100e-9)

    # Power
    v_cc = create_component("source.vdc", "cc")

    circuit.add(l_m, c_m, r_m, c_0, l_tank, c_tank_comp, c_1, c_2)
    circuit.add(q1, r_b1, r_b2, r_e, r_c, c_e, v_cc)

    # Nets
    vcc = Net("vcc")
    v_base = Net("v_base")
    v_coll = Net("v_coll")
    v_emit = Net("v_emit")
    v_xtal_mid = Net("v_xtal_mid")
    v_xtal_out = Net("v_xtal_out")

    # Power
    circuit.connect(v_cc.ports[0], vcc)
    circuit.connect(v_cc.ports[1], GND)

    # Bias
    circuit.connect(r_b1.ports[0], vcc)
    circuit.connect(r_b1.ports[1], v_base)
    circuit.connect(r_b2.ports[0], v_base)
    circuit.connect(r_b2.ports[1], GND)

    # Transistor
    circuit.connect(q1.ports[0], v_coll)
    circuit.connect(q1.ports[1], v_base)
    circuit.connect(q1.ports[2], v_emit)

    # Emitter
    circuit.connect(r_e.ports[0], v_emit)
    circuit.connect(r_e.ports[1], GND)
    circuit.connect(c_e.ports[0], v_emit)
    circuit.connect(c_e.ports[1], GND)

    # Collector through tank
    circuit.connect(l_tank.ports[0], vcc)
    circuit.connect(l_tank.ports[1], v_coll)
    circuit.connect(c_tank_comp.ports[0], vcc)
    circuit.connect(c_tank_comp.ports[1], v_coll)

    # Crystal from collector to base feedback
    circuit.connect(r_m.ports[0], v_coll)
    circuit.connect(r_m.ports[1], v_xtal_mid)
    circuit.connect(l_m.ports[0], v_xtal_mid)
    circuit.connect(l_m.ports[1], Net("v_xtal_mid2"))
    circuit.connect(c_m.ports[0], Net("v_xtal_mid2"))
    circuit.connect(c_m.ports[1], v_xtal_out)
    circuit.connect(c_0.ports[0], v_coll)
    circuit.connect(c_0.ports[1], v_xtal_out)

    # Load caps and feedback
    circuit.connect(c_1.ports[0], v_xtal_out)
    circuit.connect(c_1.ports[1], GND)
    circuit.connect(c_2.ports[0], v_xtal_out)
    circuit.connect(c_2.ports[1], v_base)

    return circuit, f_overtone, L_tank, C_tank


def main():
    """Demonstrate crystal oscillator circuits."""
    print("=" * 60)
    print("Crystal Oscillator (Pierce Configuration)")
    print("=" * 60)

    f_crystal = 10e6  # 10 MHz

    circuit1, C1, C2, Lm, Cm, Rm, C0 = build_pierce_oscillator(f_crystal)
    circuit2 = build_colpitts_crystal_oscillator(f_crystal)
    circuit3, f_ot, L_t, C_t = build_overtone_crystal_oscillator(f_crystal)

    print(f"""
   Crystal Oscillator - Precision Frequency Reference

   Crystal Model (Equivalent Circuit):

         C0 (parallel cap)
      ┌────────||────────┐
      │                  │
   ───┴──[Rm]──[Lm]──||──┴───
             series    Cm

   Crystal Parameters (10 MHz):
   - Lm = {Lm*1e3:.1f} mH (motional inductance)
   - Cm = {Cm*1e15:.1f} fF (motional capacitance)
   - Rm = {Rm} Ω (motional resistance/ESR)
   - C0 = {C0*1e12:.1f} pF (parallel capacitance)

   1. Pierce Oscillator:

         ┌────[Crystal]────┐
         │                 │
      ───┴──[Inverter]─────┴───
         │                 │
        [C1]             [C2]
         │                 │
        GND               GND

   Load capacitors: C1 = C2 = {C1*1e12:.0f} pF
   Load capacitance: CL = (C1×C2)/(C1+C2) = {C1*C2/(C1+C2)*1e12:.1f} pF

   Frequency pulling: Δf/f ≈ Cm / (2×CL)

   2. Colpitts Crystal Oscillator:

   Uses crystal in place of inductor in Colpitts topology.
   Crystal operates near series resonance.

   3. Overtone Oscillator:

   Target: {f_ot/1e6:.0f} MHz (3rd overtone of {f_crystal/1e6:.0f} MHz)
   Tank L = {L_t*1e6:.2f} µH
   Tank C = {C_t*1e12:.1f} pF

   Crystal Specifications:
   ┌───────────────────┬────────────────────────────┐
   │ Frequency         │ Marked (series or parallel)│
   │ Frequency Tol.    │ ±10 to ±100 ppm           │
   │ Load Capacitance  │ CL for specified freq     │
   │ ESR               │ Max series resistance     │
   │ Drive Level       │ Max power dissipation     │
   │ Q Factor          │ Typically 10,000-100,000  │
   │ Aging             │ ppm/year drift            │
   │ Temp Coefficient  │ ppm/°C (cut dependent)    │
   └───────────────────┴────────────────────────────┘

   Crystal Cuts:
   - AT-cut: Most common, good temp stability
   - BT-cut: Higher frequency capability
   - SC-cut: Superior stability (OCXO)

   Frequency Stability:
   - Basic crystal: ±50 ppm
   - TCXO: ±1 ppm
   - OCXO: ±0.01 ppm
   - Atomic: ±0.00001 ppm
""")

    print("   Pierce Oscillator Netlist:")
    print(circuit1.build_netlist())

    result1 = circuit1.validate()
    result2 = circuit2.validate()
    result3 = circuit3.validate()
    print(f"\n   Pierce Validation: {'VALID' if result1.is_valid else 'INVALID'}")
    print(f"   Colpitts Crystal Validation: {'VALID' if result2.is_valid else 'INVALID'}")
    print(f"   Overtone Validation: {'VALID' if result3.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
