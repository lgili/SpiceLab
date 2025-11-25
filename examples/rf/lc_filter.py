"""LC Filters (Lowpass, Highpass, Bandpass)

LC filters are passive filters using inductors and capacitors.
They handle higher power than active filters and work at RF
frequencies where op-amps don't.

Run: python examples/rf/lc_filter.py
"""

import math

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Inductor, Resistor, Vac
from spicelab.core.net import GND, Net


def build_butterworth_lowpass(
    fc: float,
    z0: float = 50,
    order: int = 3,
) -> Circuit:
    """Build a Butterworth lowpass filter.

    Butterworth has maximally flat passband response.

    Args:
        fc: Cutoff frequency in Hz
        z0: Characteristic impedance
        order: Filter order (1-5)

    Returns:
        Circuit with Butterworth lowpass filter
    """
    circuit = Circuit(f"butterworth_lowpass_n{order}")

    omega_c = 2 * math.pi * fc

    # Butterworth normalized g-values (prototype values)
    # g[k] for k = 1 to n
    g_values = {
        1: [2.0],
        2: [1.4142, 1.4142],
        3: [1.0, 2.0, 1.0],
        4: [0.7654, 1.8478, 1.8478, 0.7654],
        5: [0.618, 1.618, 2.0, 1.618, 0.618],
    }

    if order not in g_values:
        order = 3

    g = g_values[order]

    # Components
    v_in = Vac("in", ac_mag=1.0)
    r_source = Resistor("s", resistance=z0)
    r_load = Resistor("l", resistance=z0)

    circuit.add(v_in, r_source, r_load)

    # Denormalize: L = g × Z0 / ωc, C = g / (Z0 × ωc)
    # Odd elements are series L (for ladder starting with L)
    # Even elements are shunt C

    components = []
    for i, gk in enumerate(g):
        if i % 2 == 0:  # Series inductor
            L = gk * z0 / omega_c
            comp = Inductor(f"{i+1}", inductance=L)
            components.append(("L", comp, L))
        else:  # Shunt capacitor
            C = gk / (z0 * omega_c)
            comp = Capacitor(f"{i+1}", capacitance=C)
            components.append(("C", comp, C))
        circuit.add(comp)

    # Build network
    vin = Net("vin")
    prev_net = Net("v_after_rs")

    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)
    circuit.connect(r_source.ports[0], vin)
    circuit.connect(r_source.ports[1], prev_net)

    for i, (comp_type, comp, _) in enumerate(components):
        if comp_type == "L":  # Series
            next_net = Net(f"v_{i+1}")
            circuit.connect(comp.ports[0], prev_net)
            circuit.connect(comp.ports[1], next_net)
            prev_net = next_net
        else:  # Shunt
            circuit.connect(comp.ports[0], prev_net)
            circuit.connect(comp.ports[1], GND)

    # Load
    circuit.connect(r_load.ports[0], prev_net)
    circuit.connect(r_load.ports[1], GND)

    return circuit, components


def build_chebyshev_lowpass(
    fc: float,
    z0: float = 50,
    ripple_db: float = 0.5,
) -> Circuit:
    """Build a 3rd order Chebyshev lowpass filter.

    Chebyshev has steeper rolloff than Butterworth but has
    passband ripple.

    Args:
        fc: Cutoff frequency in Hz
        z0: Characteristic impedance
        ripple_db: Passband ripple in dB

    Returns:
        Circuit with Chebyshev lowpass filter
    """
    circuit = Circuit("chebyshev_lowpass_n3")

    omega_c = 2 * math.pi * fc

    # 3rd order 0.5dB ripple Chebyshev g-values
    # For other ripples/orders, use tables or calculate
    if ripple_db <= 0.1:
        g = [0.6292, 0.9703, 0.6292]
    elif ripple_db <= 0.5:
        g = [1.5963, 1.0967, 1.5963]
    else:  # 1dB
        g = [2.0236, 0.9941, 2.0236]

    # Components
    v_in = Vac("in", ac_mag=1.0)
    r_source = Resistor("s", resistance=z0)
    r_load = Resistor("l", resistance=z0)

    # Denormalize
    L1 = g[0] * z0 / omega_c
    C2 = g[1] / (z0 * omega_c)
    L3 = g[2] * z0 / omega_c

    l1 = Inductor("1", inductance=L1)
    c2 = Capacitor("2", capacitance=C2)
    l3 = Inductor("3", inductance=L3)

    circuit.add(v_in, r_source, r_load, l1, c2, l3)

    # Nets
    vin = Net("vin")
    v1 = Net("v1")
    v2 = Net("v2")
    vout = Net("vout")

    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)
    circuit.connect(r_source.ports[0], vin)
    circuit.connect(r_source.ports[1], v1)

    circuit.connect(l1.ports[0], v1)
    circuit.connect(l1.ports[1], v2)

    circuit.connect(c2.ports[0], v2)
    circuit.connect(c2.ports[1], GND)

    circuit.connect(l3.ports[0], v2)
    circuit.connect(l3.ports[1], vout)

    circuit.connect(r_load.ports[0], vout)
    circuit.connect(r_load.ports[1], GND)

    return circuit, L1, C2, L3, ripple_db


def build_bandpass_filter(
    f_center: float,
    bandwidth: float,
    z0: float = 50,
) -> Circuit:
    """Build a bandpass filter from lowpass prototype.

    Uses lowpass to bandpass transformation on a 2nd order
    prototype.

    Args:
        f_center: Center frequency in Hz
        bandwidth: 3dB bandwidth in Hz
        z0: Characteristic impedance

    Returns:
        Circuit with bandpass filter
    """
    circuit = Circuit("lc_bandpass")

    omega_0 = 2 * math.pi * f_center
    delta = bandwidth / f_center  # Fractional bandwidth

    # Start from 2nd order Butterworth LP prototype
    g1 = 1.4142  # Series element
    g2 = 1.4142  # Shunt element

    # LP to BP transformation:
    # Series L → series LC
    # Shunt C → parallel LC

    # Series LC (from g1)
    L_s = g1 * z0 / (delta * omega_0)
    C_s = delta / (g1 * z0 * omega_0)

    # Parallel LC (from g2)
    L_p = delta * z0 / (g2 * omega_0)
    C_p = g2 / (delta * z0 * omega_0)

    # Components
    v_in = Vac("in", ac_mag=1.0)
    r_source = Resistor("s", resistance=z0)
    r_load = Resistor("l", resistance=z0)

    l_s = Inductor("s", inductance=L_s)
    c_s = Capacitor("s", capacitance=C_s)
    l_p = Inductor("p", inductance=L_p)
    c_p = Capacitor("p", capacitance=C_p)

    circuit.add(v_in, r_source, r_load, l_s, c_s, l_p, c_p)

    # Nets
    vin = Net("vin")
    v1 = Net("v1")
    v2 = Net("v2")

    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)
    circuit.connect(r_source.ports[0], vin)
    circuit.connect(r_source.ports[1], v1)

    # Series LC
    circuit.connect(l_s.ports[0], v1)
    circuit.connect(l_s.ports[1], v2)
    circuit.connect(c_s.ports[0], v1)
    circuit.connect(c_s.ports[1], v2)

    # Parallel LC to ground
    circuit.connect(l_p.ports[0], v2)
    circuit.connect(l_p.ports[1], GND)
    circuit.connect(c_p.ports[0], v2)
    circuit.connect(c_p.ports[1], GND)

    # Load
    circuit.connect(r_load.ports[0], v2)
    circuit.connect(r_load.ports[1], GND)

    return circuit, L_s, C_s, L_p, C_p, f_center, bandwidth


def build_highpass_filter(
    fc: float,
    z0: float = 50,
) -> Circuit:
    """Build a 3rd order Butterworth highpass filter.

    Derived from lowpass by swapping L↔C and inverting values.

    Args:
        fc: Cutoff frequency in Hz
        z0: Characteristic impedance

    Returns:
        Circuit with highpass filter
    """
    circuit = Circuit("butterworth_highpass_n3")

    omega_c = 2 * math.pi * fc

    # 3rd order Butterworth g-values
    g = [1.0, 2.0, 1.0]

    # HP transformation: swap L and C, invert
    # LP series L → HP series C: C = 1/(g × Z0 × ωc)
    # LP shunt C → HP shunt L: L = Z0/(g × ωc)

    C1 = 1 / (g[0] * z0 * omega_c)
    L2 = z0 / (g[1] * omega_c)
    C3 = 1 / (g[2] * z0 * omega_c)

    # Components
    v_in = Vac("in", ac_mag=1.0)
    r_source = Resistor("s", resistance=z0)
    r_load = Resistor("l", resistance=z0)

    c1 = Capacitor("1", capacitance=C1)
    l2 = Inductor("2", inductance=L2)
    c3 = Capacitor("3", capacitance=C3)

    circuit.add(v_in, r_source, r_load, c1, l2, c3)

    # Nets
    vin = Net("vin")
    v1 = Net("v1")
    v2 = Net("v2")
    vout = Net("vout")

    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)
    circuit.connect(r_source.ports[0], vin)
    circuit.connect(r_source.ports[1], v1)

    # Series C1
    circuit.connect(c1.ports[0], v1)
    circuit.connect(c1.ports[1], v2)

    # Shunt L2
    circuit.connect(l2.ports[0], v2)
    circuit.connect(l2.ports[1], GND)

    # Series C3
    circuit.connect(c3.ports[0], v2)
    circuit.connect(c3.ports[1], vout)

    # Load
    circuit.connect(r_load.ports[0], vout)
    circuit.connect(r_load.ports[1], GND)

    return circuit, C1, L2, C3


def main():
    """Demonstrate LC filter circuits."""
    print("=" * 60)
    print("LC Filters (Lowpass, Highpass, Bandpass)")
    print("=" * 60)

    fc = 10e6  # 10 MHz
    f_center = 100e6  # 100 MHz
    bw = 10e6  # 10 MHz bandwidth
    z0 = 50

    circuit1, comps1 = build_butterworth_lowpass(fc, z0, 3)
    circuit2, L1_c, C2_c, L3_c, ripple = build_chebyshev_lowpass(fc, z0, 0.5)
    circuit3, Ls, Cs, Lp, Cp, f0, bw_out = build_bandpass_filter(f_center, bw, z0)
    circuit4, C1_hp, L2_hp, C3_hp = build_highpass_filter(fc, z0)

    print(f"""
   LC Filters for RF Applications

   1. Butterworth Lowpass (3rd order, fc = {fc/1e6:.0f} MHz):

   Rs ──[L1]──┬──[L3]──┬── Rl
              C2       │
              │        │
             GND      GND

   Components:""")

    for comp_type, _, value in comps1:
        if comp_type == "L":
            print(f"   - L = {value*1e9:.1f} nH")
        else:
            print(f"   - C = {value*1e12:.1f} pF")

    print(f"""
   Characteristics:
   - Maximally flat passband
   - -3dB at fc
   - -60 dB/decade rolloff (3rd order)

   2. Chebyshev Lowpass (3rd order, {ripple} dB ripple):

   Same topology, different values for steeper rolloff:
   - L1 = {L1_c*1e9:.1f} nH
   - C2 = {C2_c*1e12:.1f} pF
   - L3 = {L3_c*1e9:.1f} nH

   Characteristics:
   - Equiripple passband ({ripple} dB)
   - Steeper rolloff than Butterworth
   - More group delay variation

   3. Bandpass Filter (f0 = {f_center/1e6:.0f} MHz, BW = {bw/1e6:.0f} MHz):

   Rs ──┬──[Ls]──┬──┬── Rl
        └──[Cs]──┘  │
                   [Lp]
                    │
                   [Cp]
                    │
                   GND

   Components:
   - Ls = {Ls*1e9:.1f} nH (series)
   - Cs = {Cs*1e12:.2f} pF (series)
   - Lp = {Lp*1e9:.1f} nH (parallel)
   - Cp = {Cp*1e12:.2f} pF (parallel)

   Q = f0/BW = {f_center/bw:.1f}

   4. Highpass Filter (fc = {fc/1e6:.0f} MHz):

   Rs ──||──┬──||──┬── Rl
        C1  L2  C3
            │
           GND

   Components:
   - C1 = {C1_hp*1e12:.1f} pF
   - L2 = {L2_hp*1e9:.1f} nH
   - C3 = {C3_hp*1e12:.1f} pF

   Filter Response Comparison:
   ┌────────────┬──────────────┬────────────────────┐
   │ Type       │ Passband     │ Transition Band    │
   ├────────────┼──────────────┼────────────────────┤
   │ Butterworth│ Flat         │ Moderate rolloff   │
   │ Chebyshev  │ Ripple       │ Steep rolloff      │
   │ Bessel     │ Flat         │ Gentle rolloff     │
   │ Elliptic   │ Ripple       │ Steepest rolloff   │
   └────────────┴──────────────┴────────────────────┘

   All filters designed for {z0}Ω source and load.
""")

    print("   Butterworth LP Netlist:")
    print(circuit1.build_netlist())

    result1 = circuit1.validate()
    result2 = circuit2.validate()
    result3 = circuit3.validate()
    result4 = circuit4.validate()
    print(f"\n   Butterworth LP Validation: {'VALID' if result1.is_valid else 'INVALID'}")
    print(f"   Chebyshev LP Validation: {'VALID' if result2.is_valid else 'INVALID'}")
    print(f"   Bandpass Validation: {'VALID' if result3.is_valid else 'INVALID'}")
    print(f"   Highpass Validation: {'VALID' if result4.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
