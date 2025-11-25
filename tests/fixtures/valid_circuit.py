"""A valid circuit for testing validation CLI."""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND, Net


def create_circuit() -> Circuit:
    """Create a valid test circuit."""
    c = Circuit("test_valid")
    v = Vdc("1", 5.0)
    r = Resistor("1", resistance=1000)
    c.add(v, r)

    n1 = Net("vcc")
    c.connect(v.ports[0], n1)
    c.connect(r.ports[0], n1)
    c.connect(v.ports[1], GND)
    c.connect(r.ports[1], GND)

    return c


# Also provide as a module-level variable
circuit = create_circuit()
