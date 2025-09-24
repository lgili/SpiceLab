import re

from spicelab.core.components import MUX8, AnalogMux8
from spicelab.core.net import Port


def _net_of(port: Port) -> str:
    # simple deterministic mapping for testing
    return f"n_{port.name}"


def test_mux_static_sel() -> None:
    mux = AnalogMux8(ref="MU1", r_series=100, sel=3, enable_ports=False)
    card = mux.spice_card(_net_of)
    # header present
    assert "AnalogMux8 MU1" in card
    # selected resistor should be R MU1_3  and have value 100
    assert re.search(r"RMU1_3\s+n_in\s+n_out3\s+100", card)
    # but pattern RMU1_3 should exist pointing from n_in to n_out3
    assert "RMU1_3" in card
    # non-selected should contain off resistance
    assert "1G" in card


def test_mux_factory_and_circuit_netlist() -> None:
    mux = MUX8(r_series=100, sel=2)
    # For this test we just ensure the factory creates an AnalogMux8
    assert isinstance(mux, AnalogMux8)


def test_mux_enable_ports_and_emit_model() -> None:
    # create with enable_ports and emit_model True
    mux = AnalogMux8(ref="MU2", r_series=100, sel=None, enable_ports=True, emit_model=True)

    # build a simple net_of mapping
    def net_of(p: Port) -> str:
        return p.name

    card = mux.spice_card(net_of)
    # should contain S elements and a .model line
    assert "S" + "MU2_0" in card
    assert ".model" in card


def test_mux_as_subckt_and_validation() -> None:
    # sel range validation
    try:
        AnalogMux8(ref="MU3", sel=8)
        raise AssertionError("expected ValueError for sel out of range")
    except ValueError:
        pass

    # subckt emission
    mux = AnalogMux8(ref="MU4", r_series=100, sel=1, as_subckt=True)

    def net_of(p: Port) -> str:
        return p.name

    card = mux.spice_card(net_of)
    assert card.splitlines()[1].startswith(".subckt")
