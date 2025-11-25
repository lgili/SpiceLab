"""Tutorial 06: Diodes

This tutorial teaches you how to use diodes:
1. Basic diode behavior
2. Rectifier circuits
3. Zener voltage regulators
4. Using the component library

Run: python examples/tutorials/06_diodes.py
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Diode, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.library import create_component, get_component_spec


def demo_basic_diode():
    """Demonstrate basic diode usage."""
    print("\n1. Basic Diode")
    print("-" * 40)

    circuit = Circuit("basic_diode")

    v1 = Vdc("1", 5.0)
    r1 = Resistor("1", resistance=330)  # Current limiting
    d1 = Diode("1", model="D1N4148")

    circuit.add(v1, r1, d1)

    vcc = Net("vcc")
    vd = Net("vd")

    circuit.connect(v1.ports[0], vcc)
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r1.ports[0], vcc)
    circuit.connect(r1.ports[1], vd)
    circuit.connect(d1.ports[0], vd)  # Anode
    circuit.connect(d1.ports[1], GND)  # Cathode

    print("""
   Diode: One-way current valve

   Symbol:  Anode ─|>|─ Cathode

   Usage: Diode(ref, model="D1N4148")

   Key parameters:
   - Forward voltage (Vf): ~0.7V silicon, ~0.3V Schottky
   - Reverse breakdown: Maximum reverse voltage
   - Forward current: Maximum DC current

   Current: I = Is × (exp(V/nVt) - 1)
   Where: Vt = kT/q ≈ 26mV at room temp

   Applications:
   - Rectification
   - Protection (reverse polarity, ESD)
   - Signal clamping
   - Voltage reference
""")
    print("   Netlist:")
    print(circuit.build_netlist())
    i_led = (5.0 - 0.7) / 330
    print(f"   Expected diode current: ~{i_led*1000:.1f}mA")


def demo_half_wave_rectifier():
    """Demonstrate half-wave rectifier."""
    print("\n2. Half-Wave Rectifier")
    print("-" * 40)

    circuit = Circuit("half_wave_rect")

    # Use library component
    d1 = create_component("diode.1n4007", "1")
    r1 = Resistor("load", resistance=1000)
    c1 = Capacitor("filter", capacitance=100e-6)

    circuit.add(d1, r1, c1)

    vin = Net("vin")
    vout = Net("vout")

    # D1: vin -> vout (only positive half)
    circuit.connect(d1.ports[0], vin)
    circuit.connect(d1.ports[1], vout)
    # Load and filter in parallel
    circuit.connect(r1.ports[0], vout)
    circuit.connect(r1.ports[1], GND)
    circuit.connect(c1.ports[0], vout)
    circuit.connect(c1.ports[1], GND)

    print("""
   Half-Wave Rectifier

   Circuit:
   AC vin ───|>|───┬─── vout (DC)
                   │
                  [C]   (filter)
                   │
                  [R]   (load)
                   │
                  GND

   Operation:
   - Positive half-cycle: Diode conducts, charges capacitor
   - Negative half-cycle: Diode blocks, capacitor supplies load

   Characteristics:
   - Output ≈ Vpeak - Vf (diode drop)
   - Ripple frequency = AC frequency (60Hz for 60Hz input)
   - Ripple voltage: ΔV ≈ I_load / (f × C)

   Library diode used: 1N4007 (1A, 1000V rectifier)
""")
    spec = get_component_spec("diode.1n4007")
    print(f"   1N4007 description: {spec.metadata.get('description', 'N/A')}")
    print("   Netlist:")
    print(circuit.build_netlist())


def demo_full_wave_bridge():
    """Demonstrate full-wave bridge rectifier."""
    print("\n3. Full-Wave Bridge Rectifier")
    print("-" * 40)

    circuit = Circuit("bridge_rect")

    # Four diodes for bridge
    d1 = create_component("diode.1n4007", "1")
    d2 = create_component("diode.1n4007", "2")
    d3 = create_component("diode.1n4007", "3")
    d4 = create_component("diode.1n4007", "4")
    r1 = Resistor("load", resistance=1000)

    circuit.add(d1, d2, d3, d4, r1)

    ac_hot = Net("ac_hot")
    ac_neutral = Net("ac_neutral")
    dc_plus = Net("dc_plus")

    # Bridge configuration
    circuit.connect(d1.ports[0], ac_hot)
    circuit.connect(d1.ports[1], dc_plus)
    circuit.connect(d2.ports[0], ac_neutral)
    circuit.connect(d2.ports[1], dc_plus)
    circuit.connect(d3.ports[0], GND)
    circuit.connect(d3.ports[1], ac_hot)
    circuit.connect(d4.ports[0], GND)
    circuit.connect(d4.ports[1], ac_neutral)
    # Load
    circuit.connect(r1.ports[0], dc_plus)
    circuit.connect(r1.ports[1], GND)

    print("""
   Full-Wave Bridge Rectifier

   Circuit:
              D1          D2
   AC+ ───|>|───┬───|<|─── AC-
                │
              DC+ (output)
                │
              [R_load]
                │
   AC+ ───|<|───┴───|>|─── AC-
              D3          D4
                │
               GND

   Operation:
   - Positive half: D1,D4 conduct; D2,D3 block
   - Negative half: D2,D3 conduct; D1,D4 block
   - Both halves used → better efficiency

   Advantages over half-wave:
   - Double ripple frequency (120Hz for 60Hz input)
   - Smaller filter capacitor needed
   - Better transformer utilization
   - Output ≈ Vpeak - 2×Vf (two diode drops)
""")
    print("   Netlist:")
    print(circuit.build_netlist())


def demo_zener_regulator():
    """Demonstrate Zener voltage regulator."""
    print("\n4. Zener Voltage Regulator")
    print("-" * 40)

    circuit = Circuit("zener_reg")

    v1 = Vdc("1", 12.0)
    r1 = Resistor("1", resistance=470)  # Current limiting
    # Zener diode (5.1V) - connected in reverse
    d1 = create_component("diode.1n4733a", "1")
    r_load = Resistor("load", resistance=1000)

    circuit.add(v1, r1, d1, r_load)

    vin = Net("vin")
    vout = Net("vout")

    circuit.connect(v1.ports[0], vin)
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], vout)
    # Zener in reverse (cathode to positive)
    circuit.connect(d1.ports[1], vout)  # Cathode
    circuit.connect(d1.ports[0], GND)  # Anode
    circuit.connect(r_load.ports[0], vout)
    circuit.connect(r_load.ports[1], GND)

    print("""
   Zener Voltage Regulator

   Circuit:
   Vin ──[R]──┬── Vout (regulated)
              │
             [Dz]  (Zener, reversed)
              │
             GND

   Operation:
   - Zener operates in reverse breakdown
   - Vout ≈ Vz (Zener voltage) when properly biased
   - Excess current flows through Zener

   Design equations:
   - R = (Vin - Vz) / (Iz + Iload)
   - Iz_min ≈ 5-10mA for regulation
   - Power: Pz = Vz × Iz

   1N4733A specifications:
   - Vz = 5.1V
   - Power = 1W
   - Tolerance = 5%
""")
    spec = get_component_spec("diode.1n4733a")
    print(f"   Zener description: {spec.metadata.get('description', 'N/A')}")
    print("   Netlist:")
    print(circuit.build_netlist())
    i_total = (12.0 - 5.1) / 470
    i_load = 5.1 / 1000
    i_zener = i_total - i_load
    print("   Expected Vout: ~5.1V")
    print(f"   Zener current: ~{i_zener*1000:.1f}mA")


def main():
    """Demonstrate diode circuits."""
    print("=" * 60)
    print("Tutorial 06: Diodes")
    print("=" * 60)

    demo_basic_diode()
    demo_half_wave_rectifier()
    demo_full_wave_bridge()
    demo_zener_regulator()

    print("\n" + "=" * 60)
    print("Summary: Diode Types in Library")
    print("=" * 60)
    print("""
   | Type       | Example    | Vf     | Use Case              |
   |------------|------------|--------|------------------------|
   | Signal     | 1N4148     | 0.7V   | Fast switching        |
   | Rectifier  | 1N4007     | 1.0V   | Power rectification   |
   | Schottky   | 1N5819     | 0.3V   | Low-loss, high-freq   |
   | Zener      | 1N4733A    | 5.1V   | Voltage reference     |
   | LED        | LED_RED    | 1.8V   | Light emission        |
""")


if __name__ == "__main__":
    main()
