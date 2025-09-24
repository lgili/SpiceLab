from pathlib import Path

from spicelab.core.circuit import Circuit
from spicelab.core.components import R, V
from spicelab.core.net import GND, Net


def test_netlist_roundtrip(tmp_path: Path) -> None:
    c = Circuit(name="roundtrip")
    r1 = R(100)
    c.add(r1)
    # create a named Net and connect resistor between that net and ground
    n1 = Net("n1")
    c.connect(r1.ports[0], n1)
    c.connect(r1.ports[1], GND)
    # easier: use manual net names via Net objects by connecting to Net(name)
    # Instead, rely on build_netlist to generate names

    # add a DC source
    v1 = V(5)
    c.add(v1)

    # connect v1 between the same node and ground
    # We'll just build and save the netlist and ensure from_netlist parses it
    c.connect(v1.ports[0], n1)
    c.connect(v1.ports[1], GND)
    path = tmp_path / "test_net.net"
    c.save_netlist(str(path))

    loaded = Circuit.from_netlist(str(path))
    # The loaded netlist should build without error
    nl_loaded = loaded.build_netlist()
    assert ".end" in nl_loaded
