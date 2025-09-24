import pytest
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor


def test_unconnected_raises() -> None:
    c = Circuit("bad")
    R1 = Resistor("1", "1k")
    c.add(R1)
    with pytest.raises(ValueError):
        _ = c.build_netlist()
