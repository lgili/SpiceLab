"""An invalid circuit for testing validation CLI (has floating nodes)."""

from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor
from spicelab.core.net import Net


def create_circuit() -> Circuit:
    """Create an invalid test circuit with floating nodes."""
    c = Circuit("test_invalid")
    r1 = Resistor("1", resistance=1000)
    r2 = Resistor("2", resistance=2000)
    c.add(r1, r2)

    # Each resistor on separate floating nets (no GND, no common nodes)
    c.connect(r1.ports[0], Net("a"))
    c.connect(r1.ports[1], Net("b"))
    c.connect(r2.ports[0], Net("c"))
    c.connect(r2.ports[1], Net("d"))

    return c


circuit = create_circuit()
