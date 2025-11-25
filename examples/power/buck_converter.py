"""Buck Converter (Step-Down)

A synchronous buck converter topology for efficient voltage step-down.

Run: python examples/power/buck_converter.py
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Inductor, Resistor, Vdc, Vpulse
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def build_buck_converter(
    v_in: float = 12.0,
    v_out: float = 5.0,
    i_out: float = 1.0,
    f_sw: float = 100e3,
) -> Circuit:
    """Build a synchronous buck converter.

    Args:
        v_in: Input voltage
        v_out: Desired output voltage
        i_out: Output current
        f_sw: Switching frequency

    Returns:
        Circuit with the buck converter
    """
    circuit = Circuit("buck_converter")

    # Input supply
    vin = Vdc("in", v_in)

    # Calculate duty cycle: D = Vout/Vin
    D = v_out / v_in

    # PWM signal for high-side switch
    period = 1 / f_sw
    pw = D * period

    v_pwm_h = Vpulse("pwm_h", v1=0, v2=12, td=0, tr=10e-9, tf=10e-9, pw=pw, per=period)

    # Complementary PWM for low-side (inverted)
    v_pwm_l = Vpulse("pwm_l", v1=12, v2=0, td=0, tr=10e-9, tf=10e-9, pw=pw, per=period)

    # Power MOSFETs
    m_high = create_component("mosfet.irf540n", "h")  # High-side
    m_low = create_component("mosfet.irf540n", "l")  # Low-side (sync)

    # Output inductor
    # L = (Vin - Vout) × D / (ΔI × fsw)
    # For 30% ripple: ΔI = 0.3 × Iout
    delta_i = 0.3 * i_out
    L = (v_in - v_out) * D / (delta_i * f_sw)
    l_out = Inductor("out", inductance=L)

    # Output capacitor
    # C = ΔI / (8 × fsw × ΔV)
    # For 1% ripple: ΔV = 0.01 × Vout
    delta_v = 0.01 * v_out
    C = delta_i / (8 * f_sw * delta_v)
    c_out = Capacitor("out", capacitance=C)

    # Input capacitor
    c_in = Capacitor("in", capacitance=10e-6)

    # Load
    r_load = Resistor("load", resistance=v_out / i_out)

    # Gate drive resistors
    r_gh = Resistor("gh", resistance=10)
    r_gl = Resistor("gl", resistance=10)

    circuit.add(vin, v_pwm_h, v_pwm_l, m_high, m_low, l_out, c_out, c_in, r_load, r_gh, r_gl)

    # Nets
    v_in_net = Net("v_in")
    v_sw = Net("v_sw")
    vout = Net("vout")
    v_gh = Net("v_gh")
    v_gl = Net("v_gl")

    # Input supply and capacitor
    circuit.connect(vin.ports[0], v_in_net)
    circuit.connect(vin.ports[1], GND)
    circuit.connect(c_in.ports[0], v_in_net)
    circuit.connect(c_in.ports[1], GND)

    # PWM signals
    circuit.connect(v_pwm_h.ports[0], v_gh)
    circuit.connect(v_pwm_h.ports[1], GND)
    circuit.connect(v_pwm_l.ports[0], v_gl)
    circuit.connect(v_pwm_l.ports[1], GND)

    # Gate resistors
    circuit.connect(r_gh.ports[0], v_gh)
    circuit.connect(r_gh.ports[1], Net("g_high"))
    circuit.connect(r_gl.ports[0], v_gl)
    circuit.connect(r_gl.ports[1], Net("g_low"))

    # High-side MOSFET (D, G, S)
    circuit.connect(m_high.ports[0], v_in_net)  # Drain to Vin
    circuit.connect(m_high.ports[1], Net("g_high"))  # Gate
    circuit.connect(m_high.ports[2], v_sw)  # Source to switch node

    # Low-side MOSFET (D, G, S)
    circuit.connect(m_low.ports[0], v_sw)  # Drain to switch node
    circuit.connect(m_low.ports[1], Net("g_low"))  # Gate
    circuit.connect(m_low.ports[2], GND)  # Source to GND

    # Output LC filter
    circuit.connect(l_out.ports[0], v_sw)
    circuit.connect(l_out.ports[1], vout)
    circuit.connect(c_out.ports[0], vout)
    circuit.connect(c_out.ports[1], GND)

    # Load
    circuit.connect(r_load.ports[0], vout)
    circuit.connect(r_load.ports[1], GND)

    return circuit, D, L, C


def main():
    """Demonstrate buck converter."""
    print("=" * 60)
    print("Synchronous Buck Converter")
    print("=" * 60)

    v_in = 12.0
    v_out = 5.0
    i_out = 1.0
    f_sw = 100e3

    circuit, D, L, C = build_buck_converter(v_in, v_out, i_out, f_sw)

    print(f"""
   Synchronous Buck Converter

   Circuit topology:

   Vin ──┬──[Cin]──┐
         │        │
         D (M_high)
         │        │
         ├── Vsw ─┼──[L]──┬── Vout
         │        │       │
         S        │      [Cout]
         │        │       │
         D (M_low)│      [Rload]
         │        │       │
        GND      GND     GND

   Design parameters:
   - Input: Vin = {v_in}V
   - Output: Vout = {v_out}V @ {i_out}A
   - Switching frequency: fsw = {f_sw/1e3:.0f} kHz
   - Duty cycle: D = Vout/Vin = {D:.2%}

   Component values:
   - L = {L*1e6:.1f} µH
   - Cout = {C*1e6:.1f} µF
   - Cin = 10 µF

   Operation:
   1. High-side ON: Current flows Vin → L → Load
      Inductor charges, Vsw = Vin
   2. High-side OFF, Low-side ON: Freewheel
      Inductor supplies load, Vsw = 0

   Efficiency factors:
   - Conduction loss: I²×Rds_on
   - Switching loss: 0.5×Vin×I×(tr+tf)×fsw
   - Gate drive loss: Qg×Vg×fsw
   - Typical efficiency: 85-95%
""")

    print("   Netlist:")
    print(circuit.build_netlist())

    result = circuit.validate()
    print(f"\n   Validation: {'VALID' if result.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
