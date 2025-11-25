"""Flyback Converter

An isolated flyback converter topology for isolated power supplies.

Run: python examples/power/flyback_converter.py
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Inductor, Resistor, Vdc, Vpulse
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def build_flyback_converter(
    v_in: float = 120.0,
    v_out: float = 12.0,
    n_ratio: float = 10.0,
    f_sw: float = 65e3,
) -> Circuit:
    """Build a flyback converter.

    Note: This is a simplified model using coupled inductors.
    Real flyback uses a transformer with magnetizing inductance.

    Args:
        v_in: Input voltage (rectified AC or DC)
        v_out: Desired output voltage
        n_ratio: Turns ratio (Np/Ns)
        f_sw: Switching frequency

    Returns:
        Circuit with the flyback converter
    """
    circuit = Circuit("flyback_converter")

    # Input supply (e.g., rectified line voltage)
    vin = Vdc("in", v_in)

    # Calculate duty cycle (simplified for CCM)
    # Vout = Vin × D / (n × (1-D))
    # Solving: D = n×Vout / (Vin + n×Vout)
    D = n_ratio * v_out / (v_in + n_ratio * v_out)

    # PWM signal
    period = 1 / f_sw
    pw = D * period

    v_pwm = Vpulse("pwm", v1=0, v2=15, td=0, tr=50e-9, tf=50e-9, pw=pw, per=period)

    # Primary switch MOSFET
    m_sw = create_component("mosfet.irf540n", "sw")

    # Primary side inductor (magnetizing inductance)
    # In a real flyback, this is the transformer's magnetizing inductance
    L_mag = 500e-6  # 500µH typical
    l_pri = Inductor("pri", inductance=L_mag)

    # Secondary side diode
    d_sec = create_component("diode.1n5819", "sec")

    # Secondary inductor (for simplified coupled inductor model)
    # Ls = Lp / n²
    L_sec = L_mag / (n_ratio * n_ratio)
    l_sec = Inductor("sec", inductance=L_sec)

    # Output capacitor
    c_out = Capacitor("out", capacitance=470e-6)

    # Input capacitor
    c_in = Capacitor("in", capacitance=100e-6)

    # Snubber on primary (RCD type simplified)
    r_snub = Resistor("snub", resistance=10_000)
    c_snub = Capacitor("snub", capacitance=1e-9)

    # Load
    r_load = Resistor("load", resistance=v_out / 0.5)  # 0.5A load

    # Gate resistor
    r_g = Resistor("g", resistance=10)

    circuit.add(vin, v_pwm, m_sw, l_pri, d_sec, l_sec, c_out, c_in, r_snub, c_snub, r_load, r_g)

    # Nets
    v_in_net = Net("v_in")
    v_drain = Net("v_drain")
    v_gate = Net("v_gate")
    vout = Net("vout")
    v_sec_in = Net("v_sec_in")
    gnd_sec = Net("gnd_sec")  # Isolated secondary ground

    # Input supply and capacitor
    circuit.connect(vin.ports[0], v_in_net)
    circuit.connect(vin.ports[1], GND)
    circuit.connect(c_in.ports[0], v_in_net)
    circuit.connect(c_in.ports[1], GND)

    # Primary winding
    circuit.connect(l_pri.ports[0], v_in_net)
    circuit.connect(l_pri.ports[1], v_drain)

    # Snubber across primary
    circuit.connect(r_snub.ports[0], v_in_net)
    circuit.connect(r_snub.ports[1], Net("v_snub"))
    circuit.connect(c_snub.ports[0], Net("v_snub"))
    circuit.connect(c_snub.ports[1], v_drain)

    # PWM and gate
    circuit.connect(v_pwm.ports[0], v_gate)
    circuit.connect(v_pwm.ports[1], GND)
    circuit.connect(r_g.ports[0], v_gate)
    circuit.connect(r_g.ports[1], Net("g_sw"))

    # Primary switch (D, G, S)
    circuit.connect(m_sw.ports[0], v_drain)
    circuit.connect(m_sw.ports[1], Net("g_sw"))
    circuit.connect(m_sw.ports[2], GND)

    # Secondary winding (separate ground for isolation)
    # Note: In a real transformer model, coupling would be explicit
    circuit.connect(l_sec.ports[0], gnd_sec)
    circuit.connect(l_sec.ports[1], v_sec_in)

    # Secondary diode
    circuit.connect(d_sec.ports[0], v_sec_in)
    circuit.connect(d_sec.ports[1], vout)

    # Output capacitor and load (on secondary ground)
    circuit.connect(c_out.ports[0], vout)
    circuit.connect(c_out.ports[1], gnd_sec)
    circuit.connect(r_load.ports[0], vout)
    circuit.connect(r_load.ports[1], gnd_sec)

    return circuit, D, L_mag, n_ratio


def main():
    """Demonstrate flyback converter."""
    print("=" * 60)
    print("Flyback Converter (Isolated)")
    print("=" * 60)

    v_in = 120.0
    v_out = 12.0
    n_ratio = 10.0
    f_sw = 65e3

    circuit, D, L_mag, n = build_flyback_converter(v_in, v_out, n_ratio, f_sw)

    print(f"""
   Flyback Converter

   Circuit topology (simplified):

   Primary Side              Secondary Side
   ─────────────              ──────────────
   Vin ──┬──[Cin]            ┌──|>|──┬── Vout
         │                   │   D   │
        [Lp]═════════════════[Ls]   [Cout]
         │      (coupled)          │
        [M]                       [Rload]
         │                         │
        GND                      GND_sec

   Design parameters:
   - Input: Vin = {v_in}V (rectified line)
   - Output: Vout = {v_out}V (isolated)
   - Turns ratio: n = {n_ratio}:1
   - Switching frequency: fsw = {f_sw/1e3:.0f} kHz
   - Duty cycle: D = {D:.2%}

   Component values:
   - Lmag (primary) = {L_mag*1e6:.0f} µH
   - Cout = 470 µF

   Operation:
   1. Switch ON: Primary current ramps up
      - Energy stored in magnetic field
      - Secondary diode reverse biased
   2. Switch OFF: Flyback action
      - Voltage reverses on secondary
      - Diode conducts, energy to output
      - Vdrain = Vin + Vout×n (clamped)

   Key features:
   - Galvanic isolation (safety)
   - Multiple outputs possible
   - Wide input range capability
   - Used in AC-DC adapters, chargers
""")

    print("   Netlist:")
    print(circuit.build_netlist())

    result = circuit.validate()
    print(f"\n   Validation: {'VALID' if result.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
