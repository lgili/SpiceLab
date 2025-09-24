from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND


def test_rc_netlist_builds() -> None:
    c = Circuit("rc_lowpass")
    V1 = Vdc("1", 5.0)
    R1 = Resistor("1", "1k")
    C1 = Capacitor("1", "100n")

    c.add(V1, R1, C1)
    c.connect(V1.ports[0], R1.ports[0])  # V1.p -> R1.a
    c.connect(R1.ports[1], C1.ports[0])  # R1.b -> C1.a
    c.connect(V1.ports[1], GND)  # V1.n -> 0
    c.connect(C1.ports[1], GND)  # C1.b -> 0

    netlist = c.build_netlist()
    assert "* rc_lowpass" in netlist
    assert netlist.strip().endswith(".end")
    assert any(line.startswith("V1 ") for line in netlist.splitlines())
    assert any(line.startswith("R1 ") for line in netlist.splitlines())
    assert any(line.startswith("C1 ") for line in netlist.splitlines())
