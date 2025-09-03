import pytest
from cat.utils.units import format_eng, to_float


def test_to_float_basic() -> None:
    # use approx para tolerar diferenças de representação em ponto flutuante
    assert to_float("1k") == pytest.approx(1000.0)
    assert to_float("100n") == pytest.approx(100e-9)
    assert to_float("2.2u") == pytest.approx(2.2e-6)
    assert to_float("3meg") == pytest.approx(3e6)
    assert to_float(47.0) == pytest.approx(47.0)


def test_format_eng_roundtripish() -> None:
    s = format_eng(1000.0)
    assert s.endswith("k")
    # round-trip com tolerância
    assert to_float(s) == pytest.approx(1000.0, rel=1e-12, abs=0.0)
