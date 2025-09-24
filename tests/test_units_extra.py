import pytest
from spicelab.utils.units import format_eng, to_float


def test_to_float_unknown_suffix_raises() -> None:
    with pytest.raises(ValueError):
        to_float("1x")


def test_to_float_mega_uppercase_and_micro_symbol() -> None:
    assert to_float("3MEG") == pytest.approx(3e6)
    assert to_float("2.2µ") == pytest.approx(2.2e-6)


def test_format_eng_edges() -> None:
    # Large and small values
    assert format_eng(1e12).endswith("t")
    assert format_eng(1e-15).endswith("f")
