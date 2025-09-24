from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND, Net


def test_circuit_summary_and_to_dot() -> None:
    c = Circuit("introspect")
    n1 = Net("n1")
    V1 = Vdc("1", 1.0)
    R1 = Resistor("1", 1000.0)
    c.add(V1, R1)
    # Intentionally leave R1.a unconnected to exercise summary warnings
    c.connect(V1.ports[0], n1)
    c.connect(V1.ports[1], GND)
    # only connect one side of R1
    c.connect(R1.ports[1], GND)

    s = c.summary()
    assert "Circuit: introspect" in s
    assert "Warnings:" in s
    # to_dot should still produce a valid DOT string (no execution of dot)
    dot = c.to_dot()
    assert dot.startswith("graph circuit {")
    assert "comp_" in dot and "net_" in dot
