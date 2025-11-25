"""Practical Differentiator

An op-amp differentiator that performs mathematical differentiation.
Includes high-frequency limiting to prevent noise amplification and
instability.

Run: python examples/signal_processing/differentiator.py
"""

import math

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vsin
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def build_ideal_differentiator(f_unity: float = 1000.0) -> Circuit:
    """Build an ideal differentiator (educational - not practical).

    WARNING: This circuit has infinite gain at high frequencies,
    amplifying noise and potentially causing oscillation.

    Args:
        f_unity: Unity gain frequency

    Returns:
        Circuit with ideal differentiator
    """
    circuit = Circuit("ideal_differentiator")

    # At f_unity: |H| = 2πfRC = 1
    # Therefore: RC = 1/(2πf)
    R = 10_000
    C = 1 / (2 * math.pi * f_unity * R)

    # Input: Sine wave to demonstrate differentiation -> cosine
    v_in = Vsin("in", "0 1.0 100")  # 100 Hz sine

    # Op-amp
    u1 = create_component("opamp.ideal", "1")

    # Components
    c1 = Capacitor("1", capacitance=C)
    r1 = Resistor("1", resistance=R)

    circuit.add(v_in, u1, c1, r1)

    # Nets
    vin = Net("vin")
    vout = Net("vout")
    v_inv = Net("v_inv")

    # Input
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)

    # Input capacitor (differentiating element)
    circuit.connect(c1.ports[0], vin)
    circuit.connect(c1.ports[1], v_inv)

    # Op-amp
    circuit.connect(u1.ports[0], v_inv)
    circuit.connect(u1.ports[1], GND)
    circuit.connect(u1.ports[2], vout)

    # Feedback resistor
    circuit.connect(r1.ports[0], v_inv)
    circuit.connect(r1.ports[1], vout)

    return circuit, R, C


def build_practical_differentiator(
    f_unity: float = 1000.0,
    f_limit: float = 10000.0,
) -> Circuit:
    """Build a practical differentiator with HF limiting.

    Adds a series resistor to limit high-frequency gain and
    prevent noise amplification and instability.

    Args:
        f_unity: Unity gain frequency
        f_limit: High-frequency limit (where gain stops increasing)

    Returns:
        Circuit with practical differentiator
    """
    circuit = Circuit("practical_differentiator")

    # Design:
    # Low freq: Differentiator gain = 2πfRC
    # High freq: Gain limited to Rf/Rs
    # Crossover at f_limit: 2πf_limit*Rs*C = 1

    R_fb = 10_000
    C = 1 / (2 * math.pi * f_unity * R_fb)  # Sets unity gain freq
    R_s = 1 / (2 * math.pi * f_limit * C)  # Limits HF gain

    # Input
    v_in = Vsin("in", "0 1.0 100")

    # Op-amp
    u1 = create_component("opamp.ideal", "1")

    # Components
    c1 = Capacitor("1", capacitance=C)
    r_s = Resistor("s", resistance=R_s)  # Series limiting resistor
    r_fb = Resistor("fb", resistance=R_fb)  # Feedback resistor

    circuit.add(v_in, u1, c1, r_s, r_fb)

    # Nets
    vin = Net("vin")
    vout = Net("vout")
    v_inv = Net("v_inv")
    v_rc = Net("v_rc")

    # Input
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)

    # Input capacitor
    circuit.connect(c1.ports[0], vin)
    circuit.connect(c1.ports[1], v_rc)

    # Series resistor (HF limit)
    circuit.connect(r_s.ports[0], v_rc)
    circuit.connect(r_s.ports[1], v_inv)

    # Op-amp
    circuit.connect(u1.ports[0], v_inv)
    circuit.connect(u1.ports[1], GND)
    circuit.connect(u1.ports[2], vout)

    # Feedback resistor
    circuit.connect(r_fb.ports[0], v_inv)
    circuit.connect(r_fb.ports[1], vout)

    return circuit, R_fb, R_s, C


def build_bandlimited_differentiator(
    f_low: float = 100.0,
    f_high: float = 10000.0,
) -> Circuit:
    """Build a band-limited differentiator.

    Differentiates only within a specified frequency band.
    Acts as a constant gain amplifier at high frequencies.

    Args:
        f_low: Low frequency corner (below = attenuated)
        f_high: High frequency corner (above = constant gain)

    Returns:
        Circuit with band-limited differentiator
    """
    circuit = Circuit("bandlimited_differentiator")

    # Design for specific bandwidth
    R_fb = 10_000
    C_in = 1 / (2 * math.pi * f_low * R_fb)  # Input cap sets low corner
    R_s = R_fb * f_low / f_high  # Series R sets high corner
    C_fb = 1 / (2 * math.pi * f_high * R_fb)  # Feedback cap for stability

    # Input
    v_in = Vsin("in", "0 1.0 1000")

    # Op-amp
    u1 = create_component("opamp.ideal", "1")

    # Components
    c_in = Capacitor("in", capacitance=C_in)
    r_s = Resistor("s", resistance=R_s)
    r_fb = Resistor("fb", resistance=R_fb)
    c_fb = Capacitor("fb", capacitance=C_fb)

    circuit.add(v_in, u1, c_in, r_s, r_fb, c_fb)

    # Nets
    vin = Net("vin")
    vout = Net("vout")
    v_inv = Net("v_inv")
    v_rc = Net("v_rc")

    # Input
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)

    # Input capacitor
    circuit.connect(c_in.ports[0], vin)
    circuit.connect(c_in.ports[1], v_rc)

    # Series resistor
    circuit.connect(r_s.ports[0], v_rc)
    circuit.connect(r_s.ports[1], v_inv)

    # Op-amp
    circuit.connect(u1.ports[0], v_inv)
    circuit.connect(u1.ports[1], GND)
    circuit.connect(u1.ports[2], vout)

    # Parallel RC feedback
    circuit.connect(r_fb.ports[0], v_inv)
    circuit.connect(r_fb.ports[1], vout)
    circuit.connect(c_fb.ports[0], v_inv)
    circuit.connect(c_fb.ports[1], vout)

    return circuit, R_fb, R_s, C_in, C_fb


def main():
    """Demonstrate differentiator circuits."""
    print("=" * 60)
    print("Practical Differentiator")
    print("=" * 60)

    f_unity = 1000.0
    f_limit = 10000.0

    circuit1, R1, C1 = build_ideal_differentiator(f_unity)
    circuit2, R_fb, R_s, C2 = build_practical_differentiator(f_unity, f_limit)
    circuit3, R3, Rs3, Cin3, Cfb3 = build_bandlimited_differentiator(100, 10000)

    print(f"""
   Differentiator - Mathematical Differentiation

   1. Ideal Differentiator (NOT recommended):

              R
           ┌──[R]──┐
           │       │
   Vin ──||──┬─────┴── Vout
          C  │
            [─]
            [U1]
             │
            GND

   Transfer function: H(s) = -sRC
   Time domain: Vout = -RC × dVin/dt

   PROBLEM: Gain → ∞ as frequency → ∞
   - Amplifies high-frequency noise
   - Can cause instability/oscillation

   2. Practical Differentiator (recommended):

              Rf
           ┌──[R]──┐
           │       │
   Vin ──||─[Rs]─┬─┴── Vout
          C      │
                [─]
                [U1]
                 │
                GND

   Transfer function: H(s) = -sRfC / (1 + sRsC)

   Behavior:
   - Below f_limit: Differentiator, gain = 2πfRfC
   - Above f_limit: Inverter, gain = Rf/Rs = {R_fb/R_s:.1f}

   Component values:
   - Rf = {R_fb/1000:.1f} kΩ
   - Rs = {R_s/1000:.2f} kΩ
   - C = {C2*1e9:.2f} nF
   - Unity gain: {f_unity} Hz
   - HF limit: {f_limit} Hz

   3. Band-Limited Differentiator:

           Rf||Cf
        ┌──[Z_fb]──┐
        │          │
   Vin ─||─[Rs]──┬─┴── Vout
        Cin      │
                [U1]

   - Differentiates only between f_low and f_high
   - Low frequencies attenuated (acts as highpass)
   - High frequencies limited (acts as inverter)

   Waveform transformations:
   - Sine wave → Cosine wave (90° phase shift)
   - Triangle wave → Square wave
   - Ramp → Constant

   Applications:
   - Rate-of-change detection
   - PID controllers (D term)
   - Edge detection
   - FM demodulation
   - Velocity from position sensors
""")

    print("   Practical Differentiator Netlist:")
    print(circuit2.build_netlist())

    result1 = circuit1.validate()
    result2 = circuit2.validate()
    result3 = circuit3.validate()
    print(f"\n   Ideal Validation: {'VALID' if result1.is_valid else 'INVALID'}")
    print(f"   Practical Validation: {'VALID' if result2.is_valid else 'INVALID'}")
    print(f"   Band-Limited Validation: {'VALID' if result3.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
