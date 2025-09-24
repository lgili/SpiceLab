from spicelab.core.circuit import Circuit
from spicelab.core.components import Iac, Idc, Inductor, Vdc
from spicelab.core.net import GND, Net


def test_inductor_card_and_netlist() -> None:
    c = Circuit("rl")
    vin = Vdc("1", 1.0)
    L1 = Inductor("1", "10u")
    c.add(vin, L1)
    n1 = Net("n1")
    c.connect(vin.ports[0], n1)
    c.connect(vin.ports[1], GND)
    c.connect(L1.ports[0], n1)
    c.connect(L1.ports[1], GND)
    net = c.build_netlist()
    assert any(line.startswith("L1 ") for line in net.splitlines())


def test_current_sources_cards() -> None:
    i1 = Idc("1", "1m")
    card_dc = i1.spice_card(lambda p: "n+" if p.name == "p" else "n-")
    assert card_dc.startswith("I1 ") and " 1m" in card_dc

    i2 = Iac("2", ac_mag=2.0, ac_phase=45.0)
    card_ac = i2.spice_card(lambda p: "n+" if p.name == "p" else "n-")
    assert card_ac.startswith("I2 ") and " AC 2.0 45.0" in card_ac
