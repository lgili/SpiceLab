"""Boost Converter (Step-Up)

A boost converter topology for voltage step-up applications.

Run: python examples/power/boost_converter.py
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Inductor, Resistor, Vdc, Vpulse
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def build_boost_converter(
    v_in: float = 5.0,
    v_out: float = 12.0,
    i_out: float = 0.5,
    f_sw: float = 100e3,
) -> Circuit:
    """Build a boost converter.

    Args:
        v_in: Input voltage
        v_out: Desired output voltage
        i_out: Output current
        f_sw: Switching frequency

    Returns:
        Circuit with the boost converter
    """
    circuit = Circuit("boost_converter")

    # Input supply
    vin = Vdc("in", v_in)

    # Calculate duty cycle: D = 1 - Vin/Vout
    D = 1 - v_in / v_out

    # PWM signal
    period = 1 / f_sw
    pw = D * period

    v_pwm = Vpulse("pwm", v1=0, v2=10, td=0, tr=10e-9, tf=10e-9, pw=pw, per=period)

    # Power MOSFET
    m_sw = create_component("mosfet.irf540n", "sw")

    # Schottky diode (or use sync MOSFET)
    d_out = create_component("diode.1n5819", "out")

    # Input inductor
    # L = Vin × D / (ΔI × fsw)
    # Input current = Iout / (1-D)
    i_in = i_out / (1 - D)
    delta_i = 0.3 * i_in  # 30% ripple
    L = v_in * D / (delta_i * f_sw)
    l_in = Inductor("in", inductance=L)

    # Output capacitor
    # C = Iout × D / (fsw × ΔV)
    delta_v = 0.02 * v_out  # 2% ripple
    C = i_out * D / (f_sw * delta_v)
    c_out = Capacitor("out", capacitance=C)

    # Input capacitor
    c_in = Capacitor("in", capacitance=10e-6)

    # Load
    r_load = Resistor("load", resistance=v_out / i_out)

    # Gate resistor
    r_g = Resistor("g", resistance=10)

    circuit.add(vin, v_pwm, m_sw, d_out, l_in, c_out, c_in, r_load, r_g)

    # Nets
    v_in_net = Net("v_in")
    v_sw = Net("v_sw")
    vout = Net("vout")
    v_gate = Net("v_gate")

    # Input supply and capacitor
    circuit.connect(vin.ports[0], v_in_net)
    circuit.connect(vin.ports[1], GND)
    circuit.connect(c_in.ports[0], v_in_net)
    circuit.connect(c_in.ports[1], GND)

    # Input inductor
    circuit.connect(l_in.ports[0], v_in_net)
    circuit.connect(l_in.ports[1], v_sw)

    # PWM and gate drive
    circuit.connect(v_pwm.ports[0], v_gate)
    circuit.connect(v_pwm.ports[1], GND)
    circuit.connect(r_g.ports[0], v_gate)
    circuit.connect(r_g.ports[1], Net("g_sw"))

    # Switch MOSFET (D, G, S)
    circuit.connect(m_sw.ports[0], v_sw)  # Drain to switch node
    circuit.connect(m_sw.ports[1], Net("g_sw"))  # Gate
    circuit.connect(m_sw.ports[2], GND)  # Source to GND

    # Output diode (A, K)
    circuit.connect(d_out.ports[0], v_sw)  # Anode to switch node
    circuit.connect(d_out.ports[1], vout)  # Cathode to output

    # Output capacitor and load
    circuit.connect(c_out.ports[0], vout)
    circuit.connect(c_out.ports[1], GND)
    circuit.connect(r_load.ports[0], vout)
    circuit.connect(r_load.ports[1], GND)

    return circuit, D, L, C, i_in


def main():
    """Demonstrate boost converter."""
    print("=" * 60)
    print("Boost Converter (Step-Up)")
    print("=" * 60)

    v_in = 5.0
    v_out = 12.0
    i_out = 0.5
    f_sw = 100e3

    circuit, D, L, C, i_in = build_boost_converter(v_in, v_out, i_out, f_sw)

    print(f"""
   Boost Converter

   Circuit topology:

   Vin ──┬──[Cin]──[L]──┬──|>|──┬── Vout
         │              │   D   │
         │              │       │
        GND            [M]    [Cout]
                        │       │
                       GND    [Rload]
                               │
                              GND

   Design parameters:
   - Input: Vin = {v_in}V
   - Output: Vout = {v_out}V @ {i_out}A
   - Switching frequency: fsw = {f_sw/1e3:.0f} kHz
   - Duty cycle: D = 1 - Vin/Vout = {D:.2%}

   Component values:
   - L = {L*1e6:.1f} µH
   - Cout = {C*1e6:.1f} µF
   - Cin = 10 µF

   Current relationships:
   - Input current: Iin = Iout/(1-D) = {i_in:.2f}A
   - Power: Pin = Pout (ideal)

   Operation:
   1. Switch ON: Vin charges L, D blocks
      IL increases linearly
   2. Switch OFF: L + Vin forward biases D
      Energy transfers to output
      Vout = Vin + VL = Vin/(1-D)

   Key points:
   - Output > Input always
   - Continuous conduction mode shown
   - Input current continuous (good for batteries)
   - Output current pulsating (needs good Cout)
""")

    print("   Netlist:")
    print(circuit.build_netlist())

    result = circuit.validate()
    print(f"\n   Validation: {'VALID' if result.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
