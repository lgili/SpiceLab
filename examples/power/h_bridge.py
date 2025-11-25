"""H-Bridge Motor Driver

A full H-bridge for bidirectional DC motor control.

Run: python examples/power/h_bridge.py
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def build_h_bridge() -> Circuit:
    """Build an H-bridge motor driver.

    Returns:
        Circuit with the H-bridge
    """
    circuit = Circuit("h_bridge")

    # Power supply
    vcc = Vdc("cc", 12.0)

    # Control signals (simplified DC for demo)
    v_ah = Vdc("ah", 10.0)  # A high-side control
    v_al = Vdc("al", 0.0)  # A low-side control
    v_bh = Vdc("bh", 0.0)  # B high-side control
    v_bl = Vdc("bl", 10.0)  # B low-side control

    # Four N-channel MOSFETs (in practice, high-side would need drivers)
    m_ah = create_component("mosfet.irf540n", "ah")  # A high-side
    m_al = create_component("mosfet.irf540n", "al")  # A low-side
    m_bh = create_component("mosfet.irf540n", "bh")  # B high-side
    m_bl = create_component("mosfet.irf540n", "bl")  # B low-side

    # Gate resistors
    r_ah = Resistor("gah", resistance=100)
    r_al = Resistor("gal", resistance=100)
    r_bh = Resistor("gbh", resistance=100)
    r_bl = Resistor("gbl", resistance=100)

    # Motor (represented as resistor + inductor model)
    r_motor = Resistor("motor", resistance=5)  # Motor winding resistance

    circuit.add(
        vcc, v_ah, v_al, v_bh, v_bl, m_ah, m_al, m_bh, m_bl, r_ah, r_al, r_bh, r_bl, r_motor
    )

    # Nets
    vcc_net = Net("vcc")
    v_a = Net("v_a")
    v_b = Net("v_b")

    # Power supply
    circuit.connect(vcc.ports[0], vcc_net)
    circuit.connect(vcc.ports[1], GND)

    # Control signals and gate resistors
    circuit.connect(v_ah.ports[0], Net("ctrl_ah"))
    circuit.connect(v_ah.ports[1], GND)
    circuit.connect(r_ah.ports[0], Net("ctrl_ah"))
    circuit.connect(r_ah.ports[1], Net("g_ah"))

    circuit.connect(v_al.ports[0], Net("ctrl_al"))
    circuit.connect(v_al.ports[1], GND)
    circuit.connect(r_al.ports[0], Net("ctrl_al"))
    circuit.connect(r_al.ports[1], Net("g_al"))

    circuit.connect(v_bh.ports[0], Net("ctrl_bh"))
    circuit.connect(v_bh.ports[1], GND)
    circuit.connect(r_bh.ports[0], Net("ctrl_bh"))
    circuit.connect(r_bh.ports[1], Net("g_bh"))

    circuit.connect(v_bl.ports[0], Net("ctrl_bl"))
    circuit.connect(v_bl.ports[1], GND)
    circuit.connect(r_bl.ports[0], Net("ctrl_bl"))
    circuit.connect(r_bl.ports[1], Net("g_bl"))

    # A-leg high-side (D, G, S)
    circuit.connect(m_ah.ports[0], vcc_net)
    circuit.connect(m_ah.ports[1], Net("g_ah"))
    circuit.connect(m_ah.ports[2], v_a)

    # A-leg low-side
    circuit.connect(m_al.ports[0], v_a)
    circuit.connect(m_al.ports[1], Net("g_al"))
    circuit.connect(m_al.ports[2], GND)

    # B-leg high-side
    circuit.connect(m_bh.ports[0], vcc_net)
    circuit.connect(m_bh.ports[1], Net("g_bh"))
    circuit.connect(m_bh.ports[2], v_b)

    # B-leg low-side
    circuit.connect(m_bl.ports[0], v_b)
    circuit.connect(m_bl.ports[1], Net("g_bl"))
    circuit.connect(m_bl.ports[2], GND)

    # Motor between A and B
    circuit.connect(r_motor.ports[0], v_a)
    circuit.connect(r_motor.ports[1], v_b)

    return circuit


def main():
    """Demonstrate H-bridge motor driver."""
    print("=" * 60)
    print("H-Bridge Motor Driver")
    print("=" * 60)

    circuit = build_h_bridge()

    print("""
   H-Bridge Motor Driver

   Circuit topology:

        Vcc ───┬───────────┬─── Vcc
               │           │
              [M_AH]     [M_BH]
               │           │
        V_A ───┼───[M]─────┼─── V_B
               │           │
              [M_AL]     [M_BL]
               │           │
              GND         GND

   Control truth table:
   | AH | AL | BH | BL | Motor Action    |
   |----|----|----|----| ----------------|
   |  1 |  0 |  0 |  1 | Forward         |
   |  0 |  1 |  1 |  0 | Reverse         |
   |  0 |  0 |  0 |  0 | Coast (free)    |
   |  0 |  1 |  0 |  1 | Brake (short)   |
   |  1 |  1 |  X |  X | SHOOT-THROUGH!  |

   Current configuration (Forward):
   - AH = ON, BL = ON
   - Current: Vcc → M_AH → Motor → M_BL → GND
   - Motor voltage: +12V

   Design considerations:
   - Dead time between high/low transitions
   - High-side gate drive (bootstrap or charge pump)
   - Freewheeling diodes for inductive kickback
   - Current sensing for protection
   - PWM for speed control

   PWM control:
   - Vary duty cycle to control average voltage
   - Higher frequency = smoother torque
   - Typical: 10-20 kHz
""")

    print("   Netlist:")
    print(circuit.build_netlist())

    result = circuit.validate()
    print(f"\n   Validation: {'VALID' if result.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
