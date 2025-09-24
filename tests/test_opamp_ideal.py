from spicelab.core.circuit import Circuit
from spicelab.core.components import OA
from spicelab.core.net import Net


def test_opamp_ideal_card() -> None:
    c = Circuit("opamp_ideal")
    vinp, vinn, vout = Net("inp"), Net("inn"), Net("out")
    U = OA(1e6)
    c.add(U)
    # Connect ports: inp, inn, out
    c.connect(U.ports[0], vinp)
    c.connect(U.ports[1], vinn)
    c.connect(U.ports[2], vout)
    net = c.build_netlist()
    assert any(line.startswith("E") for line in net.splitlines())
