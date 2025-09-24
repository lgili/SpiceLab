from spicelab.core.components import Capacitor, Resistor


def test_value_si_parsing() -> None:
    r = Resistor("1", "10k")
    c = Capacitor("1", "100n")
    assert r.value_si == 10000.0
    assert c.value_si is not None
    assert abs(c.value_si - 100e-9) < 1e-15


def test_value_si_invalid() -> None:
    r = Resistor("2", "notanum")
    assert r.value_si is None
