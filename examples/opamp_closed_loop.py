from cat.core.circuit import Circuit
from cat.core.components import OpAmpIdeal, Resistor, Vdc
from cat.core.net import GND


def build_opamp_closed_loop() -> Circuit:
    """Builds a simple non-inverting op-amp closed-loop with unity gain."""
    c = Circuit("opamp_cl")
    Vin = Vdc("in", 1.0)
    Rf = Resistor("f", "0")
    Rg = Resistor("g", "0")
    op = OpAmpIdeal("1")

    c.add(Vin, Rf, Rg, op)
    # Connect input to non-inverting input
    c.connect(Vin.ports[0], op.ports[0])
    # Feedback from output to inverting input through Rf and Rg (unity for now)
    # Rf: output -> inverting input
    c.connect(op.ports[2], Rf.ports[0])
    c.connect(Rf.ports[1], op.ports[1])
    # Rg: inverting input -> ground (sets gain with Rf)
    c.connect(op.ports[1], Rg.ports[0])
    c.connect(Rg.ports[1], GND)
    # Tie input reference to ground
    c.connect(Vin.ports[1], GND)

    return c


if __name__ == "__main__":
    c = build_opamp_closed_loop()
    print(c.build_netlist())
