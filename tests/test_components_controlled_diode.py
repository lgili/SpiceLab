from spicelab.core.components import CCCS, CCVS, VCCS, VCVS, D, Diode


def _n(_: object) -> str:
    return "n1"


def test_vcvs_card() -> None:
    e = VCVS("1", 10.0)
    p, n, cp, cn = e.ports
    s = e.spice_card(lambda port: {p: "np", n: "nn", cp: "ncp", cn: "ncn"}[port])
    assert s.startswith("E1 ") and " 10.0" in s


def test_vccs_card() -> None:
    g = VCCS("2", "1e-3")
    p, n, cp, cn = g.ports
    s = g.spice_card(lambda port: {p: "np", n: "nn", cp: "ncp", cn: "ncn"}[port])
    assert s.startswith("G2 ") and " 1e-3" in s


def test_cccs_ccvs_cards() -> None:
    f = CCCS("3", "V1", 2.0)
    s1 = f.spice_card(_n)
    assert s1.startswith("F3 ") and " V1 " in s1

    h = CCVS("4", "V1", 1000.0)
    s2 = h.spice_card(_n)
    assert s2.startswith("H4 ") and " V1 " in s2


def test_diode_card_and_helper() -> None:
    d = Diode("5", "Dmod")
    s = d.spice_card(lambda _: "n1")
    assert s.startswith("D5 ") and s.endswith(" Dmod")

    d2 = D("Dmod2")
    d2.ref = "6"
    s2 = d2.spice_card(lambda _: "n2")
    assert s2.startswith("D6 ") and s2.endswith(" Dmod2")


def test_controlled_diode() -> None:
    d = D("Dcontrolled")
    d.ref = "7"
    s = d.spice_card(lambda _: "n1")
    assert s.startswith("D7 ") and s.endswith(" Dcontrolled")
