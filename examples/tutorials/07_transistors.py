"""Tutorial 07: Transistors

This tutorial teaches you how to use transistors:
1. BJT basics (NPN and PNP)
2. BJT as a switch
3. MOSFET basics
4. MOSFET as a switch

Run: python examples/tutorials/07_transistors.py
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.library import create_component, get_component_spec


def demo_npn_switch():
    """Demonstrate NPN transistor as switch."""
    print("\n1. NPN Transistor as Switch")
    print("-" * 40)

    circuit = Circuit("npn_switch")

    vcc = Vdc("cc", 12.0)
    vin = Vdc("in", 5.0)
    rb = Resistor("b", resistance=10_000)  # Base resistor
    rc = Resistor("c", resistance=1000)  # Collector resistor (load)
    q1 = create_component("bjt.2n2222", "1")

    circuit.add(vcc, vin, rb, rc, q1)

    vcc_net = Net("vcc")
    vin_net = Net("vin")
    vbase = Net("vbase")
    vcollector = Net("vcollector")

    # Power supply
    circuit.connect(vcc.ports[0], vcc_net)
    circuit.connect(vcc.ports[1], GND)

    # Input through base resistor
    circuit.connect(vin.ports[0], vin_net)
    circuit.connect(vin.ports[1], GND)
    circuit.connect(rb.ports[0], vin_net)
    circuit.connect(rb.ports[1], vbase)

    # BJT connections: C, B, E
    circuit.connect(q1.ports[0], vcollector)  # Collector
    circuit.connect(q1.ports[1], vbase)  # Base
    circuit.connect(q1.ports[2], GND)  # Emitter

    # Load resistor
    circuit.connect(rc.ports[0], vcc_net)
    circuit.connect(rc.ports[1], vcollector)

    print("""
   NPN BJT as Switch

   Circuit:
        Vcc
         │
        [Rc]
         │
         ├─── Vout
         │
   Vin ─[Rb]──┤ (Base)
              │
            ──┴── (Emitter)
              │
             GND

   Operation:
   - Vin HIGH → transistor ON (saturated) → Vout LOW
   - Vin LOW → transistor OFF (cutoff) → Vout HIGH

   Design:
   - Ib = (Vin - Vbe) / Rb ≈ (5 - 0.7) / 10k = 0.43mA
   - Ic_sat = (Vcc - Vce_sat) / Rc ≈ (12 - 0.2) / 1k = 11.8mA
   - β required = Ic / Ib = 27 (2N2222 has β > 100)

   2N2222 specifications:
   - Type: NPN
   - Ic_max: 800mA
   - hFE (β): 100-300
   - Vce_max: 40V
""")
    spec = get_component_spec("bjt.2n2222")
    print(f"   2N2222 description: {spec.metadata.get('description', 'N/A')}")
    print("   Netlist:")
    print(circuit.build_netlist())


def demo_pnp_switch():
    """Demonstrate PNP transistor as high-side switch."""
    print("\n2. PNP Transistor as High-Side Switch")
    print("-" * 40)

    circuit = Circuit("pnp_switch")

    vcc = Vdc("cc", 12.0)
    rb = Resistor("b", resistance=10_000)
    rc = Resistor("c", resistance=1000)
    q1 = create_component("bjt.2n2907", "1")

    circuit.add(vcc, rb, rc, q1)

    vcc_net = Net("vcc")
    vbase = Net("vbase")
    vload = Net("vload")

    # Power supply
    circuit.connect(vcc.ports[0], vcc_net)
    circuit.connect(vcc.ports[1], GND)

    # PNP: E, B, C (emitter to Vcc for high-side)
    circuit.connect(q1.ports[2], vcc_net)  # Emitter to Vcc
    circuit.connect(q1.ports[1], vbase)  # Base
    circuit.connect(q1.ports[0], vload)  # Collector to load

    # Base resistor to ground (to turn ON)
    circuit.connect(rb.ports[0], vbase)
    circuit.connect(rb.ports[1], GND)

    # Load to ground
    circuit.connect(rc.ports[0], vload)
    circuit.connect(rc.ports[1], GND)

    print("""
   PNP BJT as High-Side Switch

   Circuit:
        Vcc
         │
       ──┬── (Emitter)
         │
   Ctrl ─┼──┤ (Base)
         │
         ├─── Vout
         │
        [Rc]  (Load)
         │
        GND

   Operation:
   - Control LOW → transistor ON → Vout ≈ Vcc
   - Control HIGH → transistor OFF → Vout = 0

   Use case:
   - High-side switching (load connected to GND)
   - Complementary circuits with NPN

   2N2907 specifications:
   - Type: PNP
   - Ic_max: 600mA
   - hFE (β): 100-300
   - Vce_max: 60V
""")
    spec = get_component_spec("bjt.2n2907")
    print(f"   2N2907 description: {spec.metadata.get('description', 'N/A')}")
    print("   Netlist:")
    print(circuit.build_netlist())


def demo_nmos_switch():
    """Demonstrate N-channel MOSFET as switch."""
    print("\n3. N-Channel MOSFET as Switch")
    print("-" * 40)

    circuit = Circuit("nmos_switch")

    vcc = Vdc("cc", 12.0)
    vgate = Vdc("gate", 5.0)
    rd = Resistor("d", resistance=100)  # Drain resistor
    m1 = create_component("mosfet.2n7000", "1")

    circuit.add(vcc, vgate, rd, m1)

    vcc_net = Net("vcc")
    vgate_net = Net("vgate")
    vdrain = Net("vdrain")

    # Power supply
    circuit.connect(vcc.ports[0], vcc_net)
    circuit.connect(vcc.ports[1], GND)

    # Gate drive
    circuit.connect(vgate.ports[0], vgate_net)
    circuit.connect(vgate.ports[1], GND)

    # MOSFET: D, G, S
    circuit.connect(m1.ports[0], vdrain)  # Drain
    circuit.connect(m1.ports[1], vgate_net)  # Gate
    circuit.connect(m1.ports[2], GND)  # Source

    # Load resistor
    circuit.connect(rd.ports[0], vcc_net)
    circuit.connect(rd.ports[1], vdrain)

    print("""
   N-Channel MOSFET as Switch

   Circuit:
        Vcc
         │
        [Rd]
         │
         ├─── Vout
         │
   Vgate─┤├─ (Gate)
         │
        ─┴─ (Source)
         │
        GND

   Operation:
   - Vgs > Vth → MOSFET ON → Vout LOW
   - Vgs < Vth → MOSFET OFF → Vout HIGH

   Advantages over BJT:
   - Voltage-controlled (no base current needed)
   - Very high input impedance
   - Lower on-resistance for power MOSFETs
   - Faster switching

   2N7000 specifications:
   - Type: N-channel enhancement
   - Vth: 2.1V (typical)
   - Rds_on: 1.2-5Ω
   - Id_max: 200mA
   - Vds_max: 60V
""")
    spec = get_component_spec("mosfet.2n7000")
    print(f"   2N7000 description: {spec.metadata.get('description', 'N/A')}")
    print("   Netlist:")
    print(circuit.build_netlist())


def demo_power_mosfet():
    """Demonstrate power MOSFET."""
    print("\n4. Power MOSFET")
    print("-" * 40)

    circuit = Circuit("power_mosfet")

    vcc = Vdc("cc", 24.0)
    vgate = Vdc("gate", 10.0)
    rload = Resistor("load", resistance=10)  # 2.4A at 24V
    m1 = create_component("mosfet.irf540n", "1")

    circuit.add(vcc, vgate, rload, m1)

    vcc_net = Net("vcc")
    vgate_net = Net("vgate")
    vdrain = Net("vdrain")

    # Power supply
    circuit.connect(vcc.ports[0], vcc_net)
    circuit.connect(vcc.ports[1], GND)

    # Gate drive
    circuit.connect(vgate.ports[0], vgate_net)
    circuit.connect(vgate.ports[1], GND)

    # MOSFET
    circuit.connect(m1.ports[0], vdrain)
    circuit.connect(m1.ports[1], vgate_net)
    circuit.connect(m1.ports[2], GND)

    # Load
    circuit.connect(rload.ports[0], vcc_net)
    circuit.connect(rload.ports[1], vdrain)

    print("""
   Power MOSFET

   IRF540N specifications:
   - Type: N-channel enhancement
   - Vds_max: 100V
   - Id_max: 33A
   - Rds_on: 44mΩ (typical at Vgs=10V)
   - Gate charge: 71nC

   Design considerations:
   - Gate voltage: 10V for full enhancement
   - Power dissipation: P = Id² × Rds_on
   - At 2.4A: P = 2.4² × 0.044 = 0.25W

   Heat dissipation:
   - TO-220 package: ~1°C/W with heatsink
   - Junction temperature max: 175°C

   Applications:
   - Motor control
   - Switching power supplies
   - LED drivers
   - High-current switching
""")
    spec = get_component_spec("mosfet.irf540n")
    print(f"   IRF540N description: {spec.metadata.get('description', 'N/A')}")
    print("   Netlist:")
    print(circuit.build_netlist())
    p_on = (24.0 / 10) ** 2 * 0.044
    print(f"   On-state power: {p_on:.2f}W")


def main():
    """Demonstrate transistor circuits."""
    print("=" * 60)
    print("Tutorial 07: Transistors")
    print("=" * 60)

    demo_npn_switch()
    demo_pnp_switch()
    demo_nmos_switch()
    demo_power_mosfet()

    print("\n" + "=" * 60)
    print("Summary: BJT vs MOSFET")
    print("=" * 60)
    print("""
   | Feature         | BJT           | MOSFET              |
   |-----------------|---------------|---------------------|
   | Control         | Current       | Voltage             |
   | Input impedance | Low (kΩ)      | Very high (MΩ)      |
   | Saturation      | Vce_sat ~0.2V | Rds_on × Id         |
   | Speed           | Medium        | Fast                |
   | Drive power     | Continuous    | Only during switch  |
   | Best for        | Linear amps   | Switching, power    |

   Quick selection:
   - Small signal, audio: 2N2222/2N2907
   - Logic level switch: 2N7000/BS250
   - Power switching: IRF540N/IRF9540N
   - High current: TIP120 (Darlington)
""")


if __name__ == "__main__":
    main()
