import math

from spicelab.utils.e_series import enumerate_values, round_to_series


def test_enumerate_values_counts() -> None:
    vals = list(enumerate_values("E12", decade_min=0, decade_max=0))
    assert len(vals) == 12
    assert min(vals) > 0


def test_round_to_series_e24() -> None:
    # 996 should round to 1k in E24
    assert math.isclose(round_to_series(996.0, "E24"), 1000.0, rel_tol=0.01)
    # 510 should stay 510 in E24
    v = round_to_series(510.0, "E24")
    assert abs(v - 510.0) / 510.0 < 0.02
