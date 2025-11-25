"""State Variable Filter (SVF)

A universal active filter that simultaneously provides lowpass, highpass,
and bandpass outputs from a single circuit. Uses two integrators and a
summing amplifier in a feedback loop.

Run: python examples/signal_processing/state_variable_filter.py
"""

import math

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vac
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def build_state_variable_filter(
    f0: float = 1000.0,
    Q: float = 1.0,
    gain: float = 1.0,
) -> Circuit:
    """Build a state variable filter.

    The SVF provides simultaneous LP, HP, and BP outputs with
    independent control of frequency, Q, and gain.

    Args:
        f0: Center/cutoff frequency in Hz
        Q: Quality factor (0.5 = Butterworth, higher = more peaking)
        gain: Passband gain

    Returns:
        Circuit with the state variable filter
    """
    circuit = Circuit("state_variable_filter")

    # Component calculations:
    # f0 = 1 / (2 * pi * R * C)
    # Q = (R3/R4 + 1) / 3
    # Gain = R2/R1

    C = 10e-9  # 10nF - choose reasonable value
    R = 1 / (2 * math.pi * f0 * C)

    R1 = 10_000  # Input resistor
    R2 = gain * R1  # Feedback for gain
    R_int = R  # Integrator resistors

    # Q setting: Q = (1 + R3/R4) / 3
    # R3/R4 = 3*Q - 1
    R4 = 10_000
    R3 = (3 * Q - 1) * R4

    # Input source
    v_in = Vac("in", ac_mag=1.0)

    # Op-amps (summing amp + 2 integrators)
    u1 = create_component("opamp.ideal", "1")  # Summing amplifier
    u2 = create_component("opamp.ideal", "2")  # First integrator (HP->BP)
    u3 = create_component("opamp.ideal", "3")  # Second integrator (BP->LP)

    # Resistors
    r_in1 = Resistor("in1", resistance=R1)  # Input to summer
    r_fb = Resistor("fb", resistance=R2)  # Feedback to summer
    r_q1 = Resistor("q1", resistance=R3)  # Q setting
    r_q2 = Resistor("q2", resistance=R4)  # Q setting (to LP feedback)
    r_int1 = Resistor("int1", resistance=R_int)  # Integrator 1
    r_int2 = Resistor("int2", resistance=R_int)  # Integrator 2
    r_lp_fb = Resistor("lpfb", resistance=R1)  # LP feedback to summer

    # Capacitors
    c1 = Capacitor("1", capacitance=C)  # Integrator 1
    c2 = Capacitor("2", capacitance=C)  # Integrator 2

    circuit.add(v_in, u1, u2, u3)
    circuit.add(r_in1, r_fb, r_q1, r_q2, r_int1, r_int2, r_lp_fb)
    circuit.add(c1, c2)

    # Nets
    vin = Net("vin")
    v_hp = Net("v_hp")  # Highpass output (from summer)
    v_bp = Net("v_bp")  # Bandpass output (from integrator 1)
    v_lp = Net("v_lp")  # Lowpass output (from integrator 2)
    v_sum = Net("v_sum")  # Summing node
    v_int1 = Net("v_int1")  # Integrator 1 input
    v_int2 = Net("v_int2")  # Integrator 2 input

    # Input
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)

    # Input resistor to summing node
    circuit.connect(r_in1.ports[0], vin)
    circuit.connect(r_in1.ports[1], v_sum)

    # U1: Summing amplifier
    circuit.connect(u1.ports[0], v_sum)  # Inverting input
    circuit.connect(u1.ports[1], GND)  # Non-inverting to ground
    circuit.connect(u1.ports[2], v_hp)  # Output = HP

    # Feedback resistor
    circuit.connect(r_fb.ports[0], v_hp)
    circuit.connect(r_fb.ports[1], v_sum)

    # Q resistor from BP to summing node
    circuit.connect(r_q1.ports[0], v_bp)
    circuit.connect(r_q1.ports[1], v_sum)

    # LP feedback through R4 to summing node (for Q control)
    circuit.connect(r_q2.ports[0], v_lp)
    circuit.connect(r_q2.ports[1], v_sum)

    # LP feedback for DC stability
    circuit.connect(r_lp_fb.ports[0], v_lp)
    circuit.connect(r_lp_fb.ports[1], v_sum)

    # First integrator: HP -> BP
    circuit.connect(r_int1.ports[0], v_hp)
    circuit.connect(r_int1.ports[1], v_int1)
    circuit.connect(u2.ports[0], v_int1)  # Inverting input
    circuit.connect(u2.ports[1], GND)  # Non-inverting
    circuit.connect(u2.ports[2], v_bp)  # Output = BP
    circuit.connect(c1.ports[0], v_int1)
    circuit.connect(c1.ports[1], v_bp)

    # Second integrator: BP -> LP
    circuit.connect(r_int2.ports[0], v_bp)
    circuit.connect(r_int2.ports[1], v_int2)
    circuit.connect(u3.ports[0], v_int2)  # Inverting input
    circuit.connect(u3.ports[1], GND)  # Non-inverting
    circuit.connect(u3.ports[2], v_lp)  # Output = LP
    circuit.connect(c2.ports[0], v_int2)
    circuit.connect(c2.ports[1], v_lp)

    return circuit, R, C, R3, R4


def main():
    """Demonstrate state variable filter."""
    print("=" * 60)
    print("State Variable Filter (SVF)")
    print("=" * 60)

    f0 = 1000.0  # 1kHz center frequency
    Q = 1.0  # Butterworth-ish
    gain = 1.0

    circuit, R, C, R3, R4 = build_state_variable_filter(f0, Q, gain)

    print(f"""
   State Variable Filter - Universal Active Filter

   Circuit topology:

                    ┌─────[R_fb]─────┐
                    │                │
   Vin ──[R_in]──┬──┴──[U1]──HP──[R]──[U2]──BP──[R]──[U3]──LP
                 │      ▲      │        │ C        │ C   │
                 │      │      │        │          │     │
                 └[R_q1]┘      └────────┴──────────┴─────┘
                   BP              (feedback paths)
                   └─[R_q2]─LP

   Three simultaneous outputs:
   - HP (Highpass): From summing amplifier output
   - BP (Bandpass): From first integrator
   - LP (Lowpass):  From second integrator

   Design parameters:
   - Center frequency: f0 = {f0} Hz
   - Quality factor: Q = {Q}
   - Passband gain: {gain}

   Component values:
   - R_int = {R/1000:.2f} kΩ
   - C = {C*1e9:.1f} nF
   - R3 (Q set) = {R3/1000:.2f} kΩ
   - R4 (Q set) = {R4/1000:.2f} kΩ

   Transfer functions:
   - HP: H(s) = s² / (s² + s·ω0/Q + ω0²)
   - BP: H(s) = s·ω0/Q / (s² + s·ω0/Q + ω0²)
   - LP: H(s) = ω0² / (s² + s·ω0/Q + ω0²)

   Features:
   - Independent f0, Q, and gain control
   - Low sensitivity to component variations
   - Can cascade for higher-order filters
   - BP output can drive a notch summer
""")

    print("   Netlist:")
    print(circuit.build_netlist())

    result = circuit.validate()
    print(f"\n   Validation: {'VALID' if result.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
