"""Sample and Hold Circuit

A basic sample-and-hold using a MOSFET switch and op-amp buffer.

Run: python examples/analog/sample_and_hold.py
"""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.library import create_component


def build_sample_and_hold() -> Circuit:
    """Build a sample-and-hold circuit.

    Returns:
        Circuit with S/H functionality
    """
    circuit = Circuit("sample_and_hold")

    # Power supplies
    vcc = Vdc("cc", 15.0)
    vee = Vdc("ee", -15.0)

    # Control signal (sample/hold)
    v_ctrl = Vdc("ctrl", 10.0)  # HIGH = sample

    # Input buffer
    u1 = create_component("opamp.tl072", "1")

    # Switch (MOSFET)
    sw = create_component("mosfet.2n7000", "sw")

    # Hold capacitor
    c_hold = Capacitor("hold", capacitance=100e-12)  # 100pF

    # Output buffer
    u2 = create_component("opamp.tl072", "2")

    # Gate drive resistor
    r_gate = Resistor("gate", resistance=1000)

    circuit.add(vcc, vee, v_ctrl, u1, sw, c_hold, u2, r_gate)

    # Nets
    vcc_net = Net("vcc")
    vee_net = Net("vee")
    vin = Net("vin")
    v_buf1 = Net("v_buf1")
    v_hold = Net("v_hold")
    v_ctrl_net = Net("v_ctrl")
    vout = Net("vout")

    # Power supplies
    circuit.connect(vcc.ports[0], vcc_net)
    circuit.connect(vcc.ports[1], GND)
    circuit.connect(vee.ports[0], GND)
    circuit.connect(vee.ports[1], vee_net)

    # Control signal
    circuit.connect(v_ctrl.ports[0], v_ctrl_net)
    circuit.connect(v_ctrl.ports[1], GND)

    # Input buffer (unity gain follower)
    circuit.connect(u1.ports[0], v_buf1)
    circuit.connect(u1.ports[1], vin)
    circuit.connect(u1.ports[2], v_buf1)  # Unity gain
    circuit.connect(u1.ports[3], vcc_net)
    circuit.connect(u1.ports[4], vee_net)

    # Gate drive
    circuit.connect(r_gate.ports[0], v_ctrl_net)
    circuit.connect(r_gate.ports[1], Net("v_gate"))

    # MOSFET switch (D, G, S)
    circuit.connect(sw.ports[0], v_hold)  # Drain
    circuit.connect(sw.ports[1], Net("v_gate"))  # Gate
    circuit.connect(sw.ports[2], v_buf1)  # Source

    # Hold capacitor
    circuit.connect(c_hold.ports[0], v_hold)
    circuit.connect(c_hold.ports[1], GND)

    # Output buffer
    circuit.connect(u2.ports[0], vout)
    circuit.connect(u2.ports[1], v_hold)
    circuit.connect(u2.ports[2], vout)  # Unity gain
    circuit.connect(u2.ports[3], vcc_net)
    circuit.connect(u2.ports[4], vee_net)

    return circuit


def main():
    """Demonstrate sample and hold circuit."""
    print("=" * 60)
    print("Sample and Hold Circuit")
    print("=" * 60)

    circuit = build_sample_and_hold()

    print("""
   Sample and Hold Circuit

   Circuit topology:

                    MOSFET Switch
                         │
   Vin ──(+)U1──────────┤├──────┬──(+)U2── Vout
            │            │       │     │
            └────(-)─────┘      [Ch]   │
                                 │     │
                                GND    │
                                       │
                               (-)─────┘

   Control ──[Rg]── Gate

   Components:
   - U1: Input buffer (low output impedance)
   - SW: 2N7000 MOSFET switch
   - Ch: 100pF hold capacitor
   - U2: Output buffer (high input impedance)

   Operation:
   - Sample mode (ctrl HIGH): Switch ON, Ch follows Vin
   - Hold mode (ctrl LOW): Switch OFF, Ch holds voltage
   - Output buffer prevents capacitor discharge

   Key specifications:
   - Acquisition time: Time to charge Ch to new value
   - Hold droop: Voltage change during hold (leakage)
   - Aperture time: Switch opening delay
   - Feedthrough: Control signal coupling to output

   Design considerations:
   - Low Ron switch for fast acquisition
   - Low leakage switch for minimal droop
   - Small Ch = faster acquisition, more droop
   - Large Ch = slower acquisition, less droop
""")

    print("   Netlist:")
    print(circuit.build_netlist())

    result = circuit.validate()
    print(f"\n   Validation: {'VALID' if result.is_valid else 'INVALID'}")


if __name__ == "__main__":
    main()
