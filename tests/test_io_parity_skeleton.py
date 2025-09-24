from __future__ import annotations

from pathlib import Path

import pytest
from spicelab.io.readers import load_dataset


def _fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


def _approx_equal(a: float, b: float, rel: float = 0.02) -> bool:
    if b == 0:
        return abs(a) < 1e-12
    return abs(a - b) / abs(b) <= rel


@pytest.mark.parametrize(
    "ng, lt, xy",
    [
        ("rc_tran_ng.raw", "rc_tran_lt.raw", "rc_tran_xy.prn"),
    ],
)
def test_parity_rc_tran_shapes(ng: str, lt: str, xy: str) -> None:
    base = _fixtures_dir()
    ds_ng = load_dataset(base / ng, engine="ngspice")
    ds_lt = load_dataset(base / lt, engine="ltspice")
    ds_xy = load_dataset(base / xy, engine="xyce")

    # Ensure coordinate/time length parity
    n = len(ds_ng["time"].values)
    assert len(ds_lt["time"].values) == n
    assert len(ds_xy["time"].values) == n

    for var in ["V(out)", "I(R1)"]:
        assert var in ds_ng and var in ds_lt and var in ds_xy
        # Compare each sample with relaxed tolerance (2%)
        for i in range(n):
            v_ng = float(ds_ng[var].values[i])
            v_lt = float(ds_lt[var].values[i])
            v_xy = float(ds_xy[var].values[i])
            assert _approx_equal(v_lt, v_ng) or _approx_equal(v_ng, v_lt)
            assert _approx_equal(v_xy, v_ng) or _approx_equal(v_ng, v_xy)


@pytest.mark.parametrize(
    "ng, lt, xy",
    [
        ("rc_ac_ng.raw", "rc_ac_lt.raw", "rc_ac_xy.prn"),
    ],
)
def test_parity_rc_ac_shapes(ng: str, lt: str, xy: str) -> None:
    base = _fixtures_dir()
    ds_ng = load_dataset(base / ng, engine="ngspice")
    ds_lt = load_dataset(base / lt, engine="ltspice")
    ds_xy = load_dataset(base / xy, engine="xyce")

    n = len(ds_ng["freq"].values)
    assert len(ds_lt["freq"].values) == n
    assert len(ds_xy["freq"].values) == n

    for var in ["V(out)", "I(R1)"]:
        assert var in ds_ng and var in ds_lt and var in ds_xy
        for i in range(n):
            v_ng = float(ds_ng[var].values[i])
            v_lt = float(ds_lt[var].values[i])
            v_xy = float(ds_xy[var].values[i])
            assert _approx_equal(v_lt, v_ng, rel=0.15)
            assert _approx_equal(v_xy, v_ng, rel=0.15)
