"""Phase-Locked Loop (PLL) Building Blocks

PLLs lock the phase of an oscillator to a reference signal.
This example shows the basic building blocks: phase detector,
loop filter, and VCO.

Run: python examples/signal_processing/pll_basics.py
"""

import math

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vpulse
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def build_xor_phase_detector() -> Circuit:
    """Build an XOR gate phase detector.

    XOR PD produces a PWM output whose average voltage
    is proportional to the phase difference between inputs.
    Lock occurs at 90° phase difference.

    Returns:
        Circuit with XOR phase detector (conceptual)
    """
    circuit = Circuit("xor_phase_detector")

    # Reference input
    v_ref = Vpulse("ref", v1=0, v2=5, td=0, tr=1e-9, tf=1e-9, pw=0.5e-6, per=1e-6)

    # VCO input (slightly different frequency to show beat)
    v_vco = Vpulse("vco", v1=0, v2=5, td=0, tr=1e-9, tf=1e-9, pw=0.5e-6, per=1.01e-6)

    # XOR using diodes (simplified digital XOR analog)
    d1 = create_component("diode.1n4148", "1")
    d2 = create_component("diode.1n4148", "2")
    d3 = create_component("diode.1n4148", "3")
    d4 = create_component("diode.1n4148", "4")

    # Pull-up/down resistors
    r1 = Resistor("1", resistance=10_000)
    r2 = Resistor("2", resistance=10_000)

    circuit.add(v_ref, v_vco, d1, d2, d3, d4, r1, r2)

    # Nets
    v_ref_net = Net("ref")
    v_vco_net = Net("vco")
    v_pd_out = Net("pd_out")

    # Reference
    circuit.connect(v_ref.ports[0], v_ref_net)
    circuit.connect(v_ref.ports[1], GND)

    # VCO
    circuit.connect(v_vco.ports[0], v_vco_net)
    circuit.connect(v_vco.ports[1], GND)

    # XOR approximation using diode bridge
    circuit.connect(d1.ports[0], v_ref_net)
    circuit.connect(d1.ports[1], v_pd_out)

    circuit.connect(d2.ports[0], v_vco_net)
    circuit.connect(d2.ports[1], v_pd_out)

    circuit.connect(d3.ports[0], v_pd_out)
    circuit.connect(d3.ports[1], Net("mid1"))

    circuit.connect(d4.ports[0], Net("mid1"))
    circuit.connect(d4.ports[1], GND)

    circuit.connect(r1.ports[0], v_pd_out)
    circuit.connect(r1.ports[1], Net("vdd"))

    circuit.connect(r2.ports[0], v_pd_out)
    circuit.connect(r2.ports[1], GND)

    # VDD source
    v_dd = create_component("source.vdc", "dd")
    circuit.add(v_dd)
    circuit.connect(v_dd.ports[0], Net("vdd"))
    circuit.connect(v_dd.ports[1], GND)

    return circuit


def build_charge_pump_pfd() -> Circuit:
    """Build a charge pump phase-frequency detector output stage.

    The charge pump converts UP/DOWN pulses from a PFD into
    current pulses that charge/discharge the loop filter.

    Returns:
        Circuit with charge pump
    """
    circuit = Circuit("charge_pump")

    # UP pulse (from PFD when ref leads VCO)
    v_up = Vpulse("up", v1=0, v2=5, td=100e-9, tr=1e-9, tf=1e-9, pw=50e-9, per=1e-6)

    # DOWN pulse (from PFD when VCO leads ref)
    v_dn = Vpulse("dn", v1=0, v2=5, td=200e-9, tr=1e-9, tf=1e-9, pw=30e-9, per=1e-6)

    # Current source transistors (simplified with resistors)
    # In real design: matched PMOS (source) and NMOS (sink)
    q_up = create_component("bjt.2n3906", "up")  # PNP for current source
    q_dn = create_component("bjt.2n3904", "dn")  # NPN for current sink

    # Current setting resistors
    r_up = Resistor("up", resistance=10_000)  # Sets Iup
    r_dn = Resistor("dn", resistance=10_000)  # Sets Idn

    # Loop filter capacitor
    c_lf = Capacitor("lf", capacitance=1e-9)

    # Power supply
    v_dd = create_component("source.vdc", "dd")

    circuit.add(v_up, v_dn, q_up, q_dn, r_up, r_dn, c_lf, v_dd)

    # Nets
    vdd = Net("vdd")
    v_up_net = Net("up")
    v_dn_net = Net("dn")
    v_ctrl = Net("ctrl")  # Output to VCO

    # Power
    circuit.connect(v_dd.ports[0], vdd)
    circuit.connect(v_dd.ports[1], GND)

    # UP pulse
    circuit.connect(v_up.ports[0], v_up_net)
    circuit.connect(v_up.ports[1], GND)

    # DOWN pulse
    circuit.connect(v_dn.ports[0], v_dn_net)
    circuit.connect(v_dn.ports[1], GND)

    # UP current source (PNP)
    circuit.connect(q_up.ports[0], v_ctrl)  # Collector
    circuit.connect(q_up.ports[1], v_up_net)  # Base (inverted: low = on)
    circuit.connect(q_up.ports[2], Net("e_up"))  # Emitter

    circuit.connect(r_up.ports[0], vdd)
    circuit.connect(r_up.ports[1], Net("e_up"))

    # DOWN current sink (NPN)
    circuit.connect(q_dn.ports[0], v_ctrl)  # Collector
    circuit.connect(q_dn.ports[1], v_dn_net)  # Base (high = on)
    circuit.connect(q_dn.ports[2], Net("e_dn"))  # Emitter

    circuit.connect(r_dn.ports[0], Net("e_dn"))
    circuit.connect(r_dn.ports[1], GND)

    # Loop filter capacitor
    circuit.connect(c_lf.ports[0], v_ctrl)
    circuit.connect(c_lf.ports[1], GND)

    return circuit


def build_loop_filter(
    f_bw: float = 10_000,
    phase_margin: float = 60,
) -> Circuit:
    """Build a second-order loop filter.

    The loop filter converts the phase detector output to
    a smooth control voltage for the VCO.

    Args:
        f_bw: Loop bandwidth in Hz
        phase_margin: Desired phase margin in degrees

    Returns:
        Circuit with loop filter
    """
    circuit = Circuit("loop_filter")

    # Second-order passive filter
    # Provides one pole at origin (integrator) and one zero for stability

    # Design for given bandwidth and phase margin
    # τ1 = R1*C1 (zero)
    # τ2 = R2*C2 (pole)
    omega_bw = 2 * math.pi * f_bw
    phi_m = math.radians(phase_margin)

    # Rule of thumb: τ1 ≈ 1/(ω_bw * tan(φ_m))
    tau1 = 1 / (omega_bw * math.tan(phi_m))
    tau2 = tau1 / 10  # Second pole at ~10x bandwidth

    C1 = 10e-9  # Choose C1
    R1 = tau1 / C1

    C2 = 1e-9  # Smaller cap for high-freq pole
    R2 = tau2 / C2

    # Input from charge pump (simplified as pulse source)
    v_in = Vpulse("in", v1=0, v2=0.1, td=0, tr=1e-9, tf=1e-9, pw=100e-9, per=1e-6)

    # Filter components
    r1 = Resistor("1", resistance=R1)
    c1 = Capacitor("1", capacitance=C1)
    r2 = Resistor("2", resistance=R2)
    c2 = Capacitor("2", capacitance=C2)

    circuit.add(v_in, r1, c1, r2, c2)

    # Nets
    vin = Net("vin")
    v_mid = Net("v_mid")
    v_ctrl = Net("v_ctrl")

    # Input
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)

    # First stage: R1 + C1 (provides zero)
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], v_mid)
    circuit.connect(c1.ports[0], v_mid)
    circuit.connect(c1.ports[1], GND)

    # Second stage: R2 + C2 (additional filtering)
    circuit.connect(r2.ports[0], v_mid)
    circuit.connect(r2.ports[1], v_ctrl)
    circuit.connect(c2.ports[0], v_ctrl)
    circuit.connect(c2.ports[1], GND)

    return circuit, R1, C1, R2, C2


def build_vco_concept() -> Circuit:
    """Build a simplified VCO concept using RC oscillator.

    Real VCOs use LC tanks or ring oscillators. This shows
    the control voltage to frequency relationship.

    Returns:
        Circuit demonstrating VCO concept
    """
    circuit = Circuit("vco_concept")

    # Control voltage (varies frequency)
    v_ctrl = create_component("source.vdc", "ctrl")

    # Op-amp for oscillator
    u1 = create_component("opamp.ideal", "1")

    # Varactor-like behavior using voltage-dependent capacitor
    # (Simplified: fixed capacitor, real VCO uses varactor)
    c_var = Capacitor("var", capacitance=100e-12)

    # Timing resistor
    r_t = Resistor("t", resistance=10_000)

    # Feedback resistors
    r_fb1 = Resistor("fb1", resistance=10_000)
    r_fb2 = Resistor("fb2", resistance=10_000)

    # JFET as voltage-controlled resistor (varies with Vctrl)
    j1 = create_component("jfet.2n5457", "1")

    circuit.add(v_ctrl, u1, c_var, r_t, r_fb1, r_fb2, j1)

    # Nets
    v_ctrl_net = Net("ctrl")
    vout = Net("vout")
    v_inv = Net("v_inv")
    v_noninv = Net("v_noninv")

    # Control voltage
    circuit.connect(v_ctrl.ports[0], v_ctrl_net)
    circuit.connect(v_ctrl.ports[1], GND)

    # JFET controlled by Vctrl
    circuit.connect(j1.ports[0], v_inv)  # Drain
    circuit.connect(j1.ports[1], v_ctrl_net)  # Gate
    circuit.connect(j1.ports[2], GND)  # Source

    # RC timing network
    circuit.connect(r_t.ports[0], vout)
    circuit.connect(r_t.ports[1], v_noninv)
    circuit.connect(c_var.ports[0], v_noninv)
    circuit.connect(c_var.ports[1], GND)

    # Op-amp oscillator
    circuit.connect(u1.ports[0], v_inv)
    circuit.connect(u1.ports[1], v_noninv)
    circuit.connect(u1.ports[2], vout)

    # Feedback for oscillation
    circuit.connect(r_fb1.ports[0], vout)
    circuit.connect(r_fb1.ports[1], v_inv)
    circuit.connect(r_fb2.ports[0], v_inv)
    circuit.connect(r_fb2.ports[1], GND)

    return circuit


def main():
    """Demonstrate PLL building blocks."""
    print("=" * 60)
    print("Phase-Locked Loop (PLL) Building Blocks")
    print("=" * 60)

    circuit1 = build_xor_phase_detector()
    circuit2 = build_charge_pump_pfd()
    circuit3, R1, C1, R2, C2 = build_loop_filter()
    circuit4 = build_vco_concept()

    print("""
   Phase-Locked Loop - Frequency/Phase Synchronization

   PLL Block Diagram:
   ┌─────────────────────────────────────────────────┐
   │                                                 │
   │  Ref ──┬──[Phase Det]──[Loop Filter]──[VCO]──┬──Out
   │        │       │                         │    │
   │        │       └─────────────────────────┘    │
   │        │              (feedback)              │
   │        └─────────[÷N]─────────────────────────┘
   │              (optional divider)
   └─────────────────────────────────────────────────┘

   1. Phase Detector (XOR type):

   Ref ──┬──[XOR]── PD_out
         │
   VCO ──┘

   - Output = PWM proportional to phase difference
   - Average voltage = Kd × Δφ
   - Lock point at 90° phase difference

   2. Charge Pump PFD:

   UP ────[Iup]────┐
                   ├── Vctrl
   DOWN ──[Idn]────┘

   - UP pulse: sources current, raises Vctrl
   - DOWN pulse: sinks current, lowers Vctrl
   - Frequency acquisition capability
   - Zero static phase error

   3. Loop Filter (2nd order):

   PD_out ──[R1]──┬──[R2]──┬── Vctrl
                  │        │
                 [C1]     [C2]
                  │        │
                 GND      GND

   Component values:
   - R1 = {R1/1000:.2f} kΩ
   - C1 = {C1*1e9:.2f} nF
   - R2 = {R2/1000:.2f} kΩ
   - C2 = {C2*1e9:.2f} nF

   4. VCO (Voltage-Controlled Oscillator):

   - Output frequency = f0 + Kvco × Vctrl
   - Kvco = frequency sensitivity (Hz/V)
   - Real implementations: LC tank, ring oscillator

   PLL Specifications:
   ┌──────────────────┬────────────────────────────┐
   │ Lock range       │ Frequency range for lock   │
   │ Capture range    │ Range for initial lock     │
   │ Loop bandwidth   │ Speed of phase tracking    │
   │ Phase margin     │ Stability measure          │
   │ Lock time        │ Time to achieve lock       │
   │ Phase noise      │ Output spectral purity     │
   └──────────────────┴────────────────────────────┘

   Applications:
   - Clock recovery (data communications)
   - Frequency synthesis (radio, instruments)
   - FM demodulation
   - Motor speed control
   - Clock multiplication/division
""")

    print("   Loop Filter Netlist:")
    print(circuit3.build_netlist())

    result1 = circuit1.validate()
    result2 = circuit2.validate()
    result3 = circuit3.validate()
    result4 = circuit4.validate()
    print(f"\n   XOR PD Validation: {'VALID' if result1.is_valid else 'INVALID'}")
    print(f"   Charge Pump Validation: {'VALID' if result2.is_valid else 'INVALID'}")
    print(f"   Loop Filter Validation: {'VALID' if result3.is_valid else 'INVALID'}")
    print(f"   VCO Concept Validation: {'VALID' if result4.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
