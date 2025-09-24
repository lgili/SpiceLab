import math

import pytest
from spicelab.utils.units import format_eng, to_float


def test_to_float_basic_suffixes() -> None:
    assert to_float("1k") == 1000.0
    assert math.isclose(to_float("100n"), 100e-9, rel_tol=0, abs_tol=1e-18)
    assert to_float("1meg") == 1e6
    assert to_float("1m") == 1e-3
    assert to_float("1u") == 1e-6
    assert to_float("1µ") == 1e-6
    assert to_float("1g") == 1e9
    assert to_float("1t") == 1e12
    assert to_float(3.14) == 3.14


def test_to_float_invalid_suffix() -> None:
    with pytest.raises(ValueError):
        to_float("1x")  # unknown suffix
    with pytest.raises(ValueError):
        to_float("foo")


def test_format_eng_edges() -> None:
    assert format_eng(0.0) == "0"
    # Around scaling thresholds
    assert format_eng(1000.0).endswith("k")
    assert format_eng(1e6).endswith("meg")
    # Round-trip-ish check: parse back magnitude ignoring suffix text
    s = format_eng(2.2e-9)
    assert "n" in s
