from cat.core.circuit import Circuit
from cat.core.components import Capacitor, Resistor, Vdc
from cat.core.net import GND


def build_rc_highpass() -> Circuit:
    """Builds a simple RC highpass circuit: V -> C -> R -> GND"""
    c = Circuit("rc_highpass")
    V1 = Vdc("1", 1.0)
    C1 = Capacitor("1", "10n")
    R1 = Resistor("1", "1k")

    c.add(V1, C1, R1)
    c.connect(V1.ports[0], C1.ports[0])
    c.connect(C1.ports[1], R1.ports[0])
    c.connect(R1.ports[1], GND)
    c.connect(V1.ports[1], GND)

    return c


if __name__ == "__main__":
    c = build_rc_highpass()
    print(c.build_netlist())
