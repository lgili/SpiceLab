"""Impedance Matching Networks (L, Pi, T)

Impedance matching is essential in RF systems for maximum power
transfer and minimizing reflections. This example shows L-section,
Pi-network, and T-network matching topologies.

Run: python examples/rf/matching_networks.py
"""

import math

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Inductor, Resistor, Vac
from spicelab.core.net import GND, Net


def build_l_match_highpass(
    z_source: float,
    z_load: float,
    f: float,
) -> Circuit:
    """Build an L-section matching network (high-pass configuration).

    Transforms z_source to z_load using series C and shunt L.

    Args:
        z_source: Source impedance (real)
        z_load: Load impedance (real)
        f: Operating frequency in Hz

    Returns:
        Circuit with L-match network
    """
    circuit = Circuit("l_match_highpass")

    # Ensure z_load > z_source (otherwise swap topology)
    if z_load < z_source:
        z_source, z_load = z_load, z_source

    omega = 2 * math.pi * f

    # Q = √(Rhigh/Rlow - 1)
    Q = math.sqrt(z_load / z_source - 1)

    # Series reactance: Xs = Q × Rlow
    # Shunt reactance: Xp = Rhigh / Q
    Xs = Q * z_source
    Xp = z_load / Q

    # High-pass: series C, shunt L
    C_series = 1 / (omega * Xs)
    L_shunt = Xp / omega

    # Components
    v_in = Vac("in", ac_mag=1.0)
    r_source = Resistor("s", resistance=z_source)
    c_series = Capacitor("match", capacitance=C_series)
    l_shunt = Inductor("match", inductance=L_shunt)
    r_load = Resistor("l", resistance=z_load)

    circuit.add(v_in, r_source, c_series, l_shunt, r_load)

    # Nets
    vin = Net("vin")
    v_after_rs = Net("v_after_rs")
    vout = Net("vout")

    # Source
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)

    # Source resistance
    circuit.connect(r_source.ports[0], vin)
    circuit.connect(r_source.ports[1], v_after_rs)

    # Series capacitor
    circuit.connect(c_series.ports[0], v_after_rs)
    circuit.connect(c_series.ports[1], vout)

    # Shunt inductor (at output)
    circuit.connect(l_shunt.ports[0], vout)
    circuit.connect(l_shunt.ports[1], GND)

    # Load
    circuit.connect(r_load.ports[0], vout)
    circuit.connect(r_load.ports[1], GND)

    return circuit, C_series, L_shunt, Q


def build_l_match_lowpass(
    z_source: float,
    z_load: float,
    f: float,
) -> Circuit:
    """Build an L-section matching network (low-pass configuration).

    Transforms z_source to z_load using series L and shunt C.

    Args:
        z_source: Source impedance (real)
        z_load: Load impedance (real)
        f: Operating frequency in Hz

    Returns:
        Circuit with L-match network
    """
    circuit = Circuit("l_match_lowpass")

    if z_load < z_source:
        z_source, z_load = z_load, z_source

    omega = 2 * math.pi * f

    Q = math.sqrt(z_load / z_source - 1)
    Xs = Q * z_source
    Xp = z_load / Q

    # Low-pass: series L, shunt C
    L_series = Xs / omega
    C_shunt = 1 / (omega * Xp)

    # Components
    v_in = Vac("in", ac_mag=1.0)
    r_source = Resistor("s", resistance=z_source)
    l_series = Inductor("match", inductance=L_series)
    c_shunt = Capacitor("match", capacitance=C_shunt)
    r_load = Resistor("l", resistance=z_load)

    circuit.add(v_in, r_source, l_series, c_shunt, r_load)

    # Nets
    vin = Net("vin")
    v_after_rs = Net("v_after_rs")
    vout = Net("vout")

    # Source
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)
    circuit.connect(r_source.ports[0], vin)
    circuit.connect(r_source.ports[1], v_after_rs)

    # Series inductor
    circuit.connect(l_series.ports[0], v_after_rs)
    circuit.connect(l_series.ports[1], vout)

    # Shunt capacitor
    circuit.connect(c_shunt.ports[0], vout)
    circuit.connect(c_shunt.ports[1], GND)

    # Load
    circuit.connect(r_load.ports[0], vout)
    circuit.connect(r_load.ports[1], GND)

    return circuit, L_series, C_shunt, Q


def build_pi_network(
    z_source: float,
    z_load: float,
    f: float,
    Q_desired: float = 5.0,
) -> Circuit:
    """Build a Pi matching network.

    Pi network offers more design flexibility through selectable Q.
    Higher Q = narrower bandwidth, more filtering.

    Args:
        z_source: Source impedance
        z_load: Load impedance
        f: Operating frequency
        Q_desired: Network Q factor

    Returns:
        Circuit with Pi network
    """
    circuit = Circuit("pi_network")

    omega = 2 * math.pi * f

    # Pi network design:
    # Virtual resistance: Rv = min(Rs, Rl) / (Q² + 1)
    R_min = min(z_source, z_load)
    Rv = R_min / (Q_desired**2 + 1)

    # Shunt element at source: Xp1 = Rs × Rv / √(Rs² - Rv²)
    # Shunt element at load: Xp2 = Rl × Rv / √(Rl² - Rv²)
    # Series element: Xs = Q × Rv

    # For simplicity, using symmetric design when Rs = Rl
    Xp1 = z_source / Q_desired
    Xp2 = z_load / Q_desired
    Xs = Q_desired * Rv * 2  # Approximate

    # Low-pass Pi: shunt C, series L, shunt C
    C1 = 1 / (omega * Xp1)
    L = Xs / omega
    C2 = 1 / (omega * Xp2)

    # Components
    v_in = Vac("in", ac_mag=1.0)
    r_source = Resistor("s", resistance=z_source)
    c1 = Capacitor("1", capacitance=C1)
    l1 = Inductor("1", inductance=L)
    c2 = Capacitor("2", capacitance=C2)
    r_load = Resistor("l", resistance=z_load)

    circuit.add(v_in, r_source, c1, l1, c2, r_load)

    # Nets
    vin = Net("vin")
    v_after_rs = Net("v_after_rs")
    vout = Net("vout")

    # Source
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)
    circuit.connect(r_source.ports[0], vin)
    circuit.connect(r_source.ports[1], v_after_rs)

    # Input shunt C
    circuit.connect(c1.ports[0], v_after_rs)
    circuit.connect(c1.ports[1], GND)

    # Series L
    circuit.connect(l1.ports[0], v_after_rs)
    circuit.connect(l1.ports[1], vout)

    # Output shunt C
    circuit.connect(c2.ports[0], vout)
    circuit.connect(c2.ports[1], GND)

    # Load
    circuit.connect(r_load.ports[0], vout)
    circuit.connect(r_load.ports[1], GND)

    return circuit, C1, L, C2, Q_desired


def build_t_network(
    z_source: float,
    z_load: float,
    f: float,
    Q_desired: float = 5.0,
) -> Circuit:
    """Build a T matching network.

    T network is the dual of Pi network. Uses two series elements
    and one shunt element.

    Args:
        z_source: Source impedance
        z_load: Load impedance
        f: Operating frequency
        Q_desired: Network Q factor

    Returns:
        Circuit with T network
    """
    circuit = Circuit("t_network")

    omega = 2 * math.pi * f

    # T network: series L, shunt C, series L
    Xs1 = Q_desired * z_source
    Xs2 = Q_desired * z_load
    Xp = (z_source + z_load) / Q_desired

    # Low-pass T: series L, shunt C, series L
    L1 = Xs1 / omega
    C = 1 / (omega * Xp)
    L2 = Xs2 / omega

    # Components
    v_in = Vac("in", ac_mag=1.0)
    r_source = Resistor("s", resistance=z_source)
    l1 = Inductor("1", inductance=L1)
    c1 = Capacitor("1", capacitance=C)
    l2 = Inductor("2", inductance=L2)
    r_load = Resistor("l", resistance=z_load)

    circuit.add(v_in, r_source, l1, c1, l2, r_load)

    # Nets
    vin = Net("vin")
    v_after_rs = Net("v_after_rs")
    v_mid = Net("v_mid")
    vout = Net("vout")

    # Source
    circuit.connect(v_in.ports[0], vin)
    circuit.connect(v_in.ports[1], GND)
    circuit.connect(r_source.ports[0], vin)
    circuit.connect(r_source.ports[1], v_after_rs)

    # Series L1
    circuit.connect(l1.ports[0], v_after_rs)
    circuit.connect(l1.ports[1], v_mid)

    # Shunt C
    circuit.connect(c1.ports[0], v_mid)
    circuit.connect(c1.ports[1], GND)

    # Series L2
    circuit.connect(l2.ports[0], v_mid)
    circuit.connect(l2.ports[1], vout)

    # Load
    circuit.connect(r_load.ports[0], vout)
    circuit.connect(r_load.ports[1], GND)

    return circuit, L1, C, L2, Q_desired


def main():
    """Demonstrate impedance matching networks."""
    print("=" * 60)
    print("Impedance Matching Networks (L, Pi, T)")
    print("=" * 60)

    z_source = 50  # 50Ω source
    z_load = 200  # 200Ω load
    f = 100e6  # 100 MHz

    circuit1, C_hp, L_hp, Q_hp = build_l_match_highpass(z_source, z_load, f)
    circuit2, L_lp, C_lp, Q_lp = build_l_match_lowpass(z_source, z_load, f)
    circuit3, C1_pi, L_pi, C2_pi, Q_pi = build_pi_network(z_source, z_load, f)
    circuit4, L1_t, C_t, L2_t, Q_t = build_t_network(z_source, z_load, f)

    print(f"""
   Impedance Matching: {z_source}Ω → {z_load}Ω at {f/1e6:.0f} MHz

   1. L-Match (High-Pass):

   Rs ──||──┬── Rl
        Cs  L
            │
           GND

   Components:
   - Cs = {C_hp*1e12:.2f} pF (series)
   - L = {L_hp*1e9:.1f} nH (shunt)
   - Q = {Q_hp:.2f}

   2. L-Match (Low-Pass):

   Rs ──[L]──┬── Rl
             Cp
             │
            GND

   Components:
   - L = {L_lp*1e9:.1f} nH (series)
   - Cp = {C_lp*1e12:.2f} pF (shunt)
   - Q = {Q_lp:.2f}

   3. Pi Network:

   Rs ──┬──[L]──┬── Rl
        C1      C2
        │       │
       GND     GND

   Components:
   - C1 = {C1_pi*1e12:.2f} pF (input shunt)
   - L = {L_pi*1e9:.1f} nH (series)
   - C2 = {C2_pi*1e12:.2f} pF (output shunt)
   - Q = {Q_pi:.1f}

   4. T Network:

   Rs ──[L1]──┬──[L2]── Rl
              C
              │
             GND

   Components:
   - L1 = {L1_t*1e9:.1f} nH (input series)
   - C = {C_t*1e12:.2f} pF (shunt)
   - L2 = {L2_t*1e9:.1f} nH (output series)
   - Q = {Q_t:.1f}

   Network Comparison:
   ┌──────────┬────────────┬────────────┬────────────┐
   │ Network  │ Complexity │ Bandwidth  │ Harmonic   │
   │          │            │            │ Filtering  │
   ├──────────┼────────────┼────────────┼────────────┤
   │ L-Match  │ Simplest   │ Widest     │ Minimal    │
   │ Pi       │ Medium     │ Adjustable │ Good (LP)  │
   │ T        │ Medium     │ Adjustable │ Good (LP)  │
   └──────────┴────────────┴────────────┴────────────┘

   Design Considerations:
   - Q factor determines bandwidth: BW = f/Q
   - Higher Q = narrower bandwidth, more selectivity
   - Low-pass versions provide harmonic filtering
   - Component Q (quality) affects loss
   - Parasitics important at high frequencies

   Applications:
   - Antenna matching
   - Amplifier input/output
   - Filter interfaces
   - Power amplifier output
""")

    print("   L-Match Low-Pass Netlist:")
    print(circuit2.build_netlist())

    result1 = circuit1.validate()
    result2 = circuit2.validate()
    result3 = circuit3.validate()
    result4 = circuit4.validate()
    print(f"\n   L-Match HP Validation: {'VALID' if result1.is_valid else 'INVALID'}")
    print(f"   L-Match LP Validation: {'VALID' if result2.is_valid else 'INVALID'}")
    print(f"   Pi Network Validation: {'VALID' if result3.is_valid else 'INVALID'}")
    print(f"   T Network Validation: {'VALID' if result4.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
