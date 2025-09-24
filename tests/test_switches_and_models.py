from spicelab.core.circuit import Circuit
from spicelab.core.components import SW, SWI, R, V
from spicelab.core.net import GND, Net


def test_voltage_switch_and_model_directive() -> None:
    c = Circuit("vsw")
    a, b, ctrlp, _ctrln = Net("a"), Net("b"), Net("cp"), Net("cn")
    V1 = V(1.0)
    S1 = SW("SWMOD")
    R1 = R("1k")
    c.add(V1, S1, R1)
    c.connect(V1.ports[0], ctrlp)
    c.connect(V1.ports[1], GND)
    p, n, cp, cn = S1.ports
    c.connect(p, a)
    c.connect(n, b)
    c.connect(cp, ctrlp)
    c.connect(cn, GND)
    # Load resistor from b to GND
    c.connect(R1.ports[0], b)
    c.connect(R1.ports[1], GND)
    # Add model directive
    c.add_directive(".model SWMOD VSWITCH(Ron=1 Roff=1e9 Von=0.5 Voff=0.4)")
    net = c.build_netlist()
    assert ".model SWMOD" in net and any(line.startswith("S") for line in net.splitlines())


def test_current_switch_and_model_directive() -> None:
    c = Circuit("isw")
    a, b = Net("a"), Net("b")
    V1 = V(1.0)
    W1 = SWI("V1", "WMOD")
    R1 = R("1k")
    c.add(V1, W1, R1)
    c.connect(V1.ports[0], a)
    c.connect(V1.ports[1], GND)
    p, n = W1.ports
    c.connect(p, a)
    c.connect(n, b)
    # Load resistor from b to GND
    c.connect(R1.ports[0], b)
    c.connect(R1.ports[1], GND)
    c.add_directive(".model WMOD ISWITCH(Ion=1m Ioff=1u Ron=1 Roff=1e9)")
    net = c.build_netlist()
    assert ".model WMOD" in net and any(line.startswith("W") for line in net.splitlines())
