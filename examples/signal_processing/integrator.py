"""Miller Integrator

An op-amp integrator that performs mathematical integration of
the input signal. Essential for analog computing, waveform
generation, and control systems.

Run: python examples/signal_processing/integrator.py
"""

import math

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vpulse
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def build_basic_integrator(
    f_unity: float = 1000.0,
) -> Circuit:
    """Build a basic Miller integrator.

    The output is the integral of the input:
    Vout = -(1/RC) ∫ Vin dt

    Args:
        f_unity: Unity gain frequency (where |H| = 1)

    Returns:
        Circuit with the integrator
    """
    circuit = Circuit("miller_integrator")

    # At f_unity: |H| = 1/(2πfRC) = 1
    # Therefore: RC = 1/(2πf)
    C = 10e-9  # 10nF
    R = 1 / (2 * math.pi * f_unity * C)

    # Input: Square wave to demonstrate integration -> triangle
    v_in = Vpulse("in", v1=-1, v2=1, td=0, tr=1e-9, tf=1e-9, pw=0.5e-3, per=1e-3)

    # Op-amp
    u1 = create_component("opamp.ideal", "1")

    # Resistor and capacitor
    r1 = Resistor("1", resistance=R)
    c1 = Capacitor("1", capacitance=C)

    circuit.add(v_in, u1, r1, c1)

    # Nets
    vin = Net("vin")
    vout = Net("vout")
    v_inv = Net("v_inv")

    # Input
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)

    # Input resistor
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], v_inv)

    # Op-amp
    circuit.connect(u1.ports[0], v_inv)  # Inverting input
    circuit.connect(u1.ports[1], GND)  # Non-inverting to GND
    circuit.connect(u1.ports[2], vout)  # Output

    # Integrating capacitor in feedback
    circuit.connect(c1.ports[0], v_inv)
    circuit.connect(c1.ports[1], vout)

    return circuit, R, C


def build_lossy_integrator(
    f_unity: float = 1000.0,
    dc_gain: float = 100.0,
) -> Circuit:
    """Build a lossy (practical) integrator.

    Adds a feedback resistor to limit DC gain and prevent
    saturation from DC offsets and bias currents.

    Args:
        f_unity: Unity gain frequency
        dc_gain: DC gain limit (Rf/Rin)

    Returns:
        Circuit with lossy integrator
    """
    circuit = Circuit("lossy_integrator")

    C = 10e-9
    R_in = 1 / (2 * math.pi * f_unity * C)
    R_fb = dc_gain * R_in  # DC gain = Rf/Rin

    # Input
    v_in = Vpulse("in", v1=-1, v2=1, td=0, tr=1e-9, tf=1e-9, pw=0.5e-3, per=1e-3)

    # Op-amp
    u1 = create_component("opamp.ideal", "1")

    # Components
    r_in = Resistor("in", resistance=R_in)
    r_fb = Resistor("fb", resistance=R_fb)
    c_fb = Capacitor("fb", capacitance=C)

    circuit.add(v_in, u1, r_in, r_fb, c_fb)

    # Nets
    vin = Net("vin")
    vout = Net("vout")
    v_inv = Net("v_inv")

    # Input
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)

    # Input resistor
    circuit.connect(r_in.ports[0], vin)
    circuit.connect(r_in.ports[1], v_inv)

    # Op-amp
    circuit.connect(u1.ports[0], v_inv)
    circuit.connect(u1.ports[1], GND)
    circuit.connect(u1.ports[2], vout)

    # Parallel RC feedback
    circuit.connect(c_fb.ports[0], v_inv)
    circuit.connect(c_fb.ports[1], vout)
    circuit.connect(r_fb.ports[0], v_inv)
    circuit.connect(r_fb.ports[1], vout)

    return circuit, R_in, R_fb, C


def build_integrator_with_reset() -> Circuit:
    """Build an integrator with reset capability.

    Uses a MOSFET switch to discharge the integrating capacitor,
    resetting the output to zero.

    Returns:
        Circuit with resettable integrator
    """
    circuit = Circuit("integrator_with_reset")

    R = 10_000
    C = 100e-9

    # Input signal
    v_in = Vpulse("in", v1=0, v2=1, td=0, tr=1e-9, tf=1e-9, pw=1e-3, per=2e-3)

    # Reset pulse (active high)
    v_reset = Vpulse("rst", v1=0, v2=5, td=1.8e-3, tr=1e-9, tf=1e-9, pw=0.1e-3, per=2e-3)

    # Op-amp
    u1 = create_component("opamp.ideal", "1")

    # Reset MOSFET
    m_rst = create_component("mosfet.2n7002", "rst")

    # Components
    r1 = Resistor("1", resistance=R)
    c1 = Capacitor("1", capacitance=C)

    circuit.add(v_in, v_reset, u1, m_rst, r1, c1)

    # Nets
    vin = Net("vin")
    vout = Net("vout")
    v_inv = Net("v_inv")
    v_rst = Net("v_rst")

    # Input
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)

    # Reset signal
    circuit.connect(v_reset.ports[0], v_rst)
    circuit.connect(v_reset.ports[1], GND)

    # Input resistor
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], v_inv)

    # Op-amp
    circuit.connect(u1.ports[0], v_inv)
    circuit.connect(u1.ports[1], GND)
    circuit.connect(u1.ports[2], vout)

    # Integrating capacitor
    circuit.connect(c1.ports[0], v_inv)
    circuit.connect(c1.ports[1], vout)

    # Reset MOSFET across capacitor (D, G, S)
    circuit.connect(m_rst.ports[0], vout)  # Drain
    circuit.connect(m_rst.ports[1], v_rst)  # Gate
    circuit.connect(m_rst.ports[2], v_inv)  # Source

    return circuit, R, C


def main():
    """Demonstrate integrator circuits."""
    print("=" * 60)
    print("Miller Integrator")
    print("=" * 60)

    f_unity = 1000.0
    dc_gain = 100.0

    circuit1, R1, C1 = build_basic_integrator(f_unity)
    circuit2, R_in, R_fb, C2 = build_lossy_integrator(f_unity, dc_gain)
    circuit3, R3, C3 = build_integrator_with_reset()

    print(f"""
   Miller Integrator - Mathematical Integration

   1. Basic (Ideal) Integrator:

              C
           ┌──||──┐
           │      │
   Vin ──[R]──┬───┴──── Vout
              │
             [─]
             [U1]
              │
             GND

   Transfer function: H(s) = -1/(sRC)
   Time domain: Vout = -(1/RC) ∫ Vin dt

   Component values:
   - R = {R1/1000:.2f} kΩ
   - C = {C1*1e9:.1f} nF
   - τ = RC = {R1*C1*1e6:.1f} µs
   - Unity gain frequency: {f_unity} Hz

   2. Lossy (Practical) Integrator:

           Rf
        ┌──[R]──┐
        │   C   │
        ├──||───┤
        │       │
   Vin ─[Rin]─┬─┴── Vout

   - Rf limits DC gain to {dc_gain}
   - Prevents saturation from offsets
   - Acts as integrator above f = 1/(2πRfC)
   - Acts as inverter below that frequency

   3. Integrator with Reset:

           C
        ┌──||──┐
        │  [M] │  <- Reset MOSFET
        │      │
   Vin ─[R]─┬──┴── Vout

   - MOSFET shorts C when reset is high
   - Allows periodic reset in control loops
   - Essential for sample-and-hold, ADCs

   Waveform transformations:
   - Square wave → Triangle wave
   - Triangle wave → Parabola
   - Sine wave → -Cosine wave

   Applications:
   - Analog computers
   - PID controllers (I term)
   - Waveform generators
   - Charge amplifiers
   - Active filters
""")

    print("   Basic Integrator Netlist:")
    print(circuit1.build_netlist())

    result1 = circuit1.validate()
    result2 = circuit2.validate()
    result3 = circuit3.validate()
    print(f"\n   Basic Validation: {'VALID' if result1.is_valid else 'INVALID'}")
    print(f"   Lossy Validation: {'VALID' if result2.is_valid else 'INVALID'}")
    print(f"   Reset Validation: {'VALID' if result3.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
