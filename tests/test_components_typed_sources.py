from spicelab.core.components import IPWL_T, ISIN_T, VPWL_T, VSIN_T


def _net_of(_: object) -> str:
    return "n1"


def test_vsin_t_card() -> None:
    v = VSIN_T(0, 1, "1k", td="10u")
    v.ref = "1"
    s = v.spice_card(_net_of)
    assert s.startswith("V1 ") and "SIN(" in s and "1k" in s and "10u" in s


def test_isin_t_card() -> None:
    i = ISIN_T("0", "1m", "1k")
    i.ref = "2"
    s = i.spice_card(_net_of)
    assert s.startswith("I2 ") and "SIN(" in s and "1m" in s


def test_vpwl_t_card() -> None:
    v = VPWL_T([(0, 0), ("1m", 5), ("2m", 0)])
    v.ref = "3"
    s = v.spice_card(_net_of)
    assert s.startswith("V3 ") and "PWL(" in s and "1m 5" in s


def test_ipwl_t_card() -> None:
    i = IPWL_T([(0, 0), ("1m", "1m")])
    i.ref = "4"
    s = i.spice_card(_net_of)
    assert s.startswith("I4 ") and "PWL(" in s and "1m 1m" in s
