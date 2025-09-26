from __future__ import annotations

import numpy as np
import pytest
import xarray as xr  # type: ignore[import-not-found]
from spicelab.analysis import GainSpec, OvershootSpec, SettlingTimeSpec, measure

pytest.importorskip("polars")


def test_measure_gain_db() -> None:
    freq = np.array([10.0, 100.0, 1_000.0])
    vout = np.array([0.1, 1.0, 10.0])
    vin = np.ones_like(vout)
    ds = xr.Dataset(
        data_vars={
            "V(out)": ("freq", vout),
            "V(in)": ("freq", vin),
        },
        coords={"freq": freq},
    )

    specs = [GainSpec(name="gain_1k", numerator="V(out)", denominator="V(in)", freq=1_000.0)]
    df = measure(ds, specs)
    assert df.shape == (1, 7)
    row = df.row(0, named=True)
    assert row["measure"] == "gain_1k"
    assert row["units"] == "dB"
    # 10/1 -> 20 dB
    assert abs(row["value"] - 20.0) < 1e-6


def test_measure_overshoot_percent() -> None:
    time = np.array([0.0, 1.0, 2.0, 3.0])
    vout = np.array([0.0, 1.2, 1.05, 1.0])
    ds = xr.Dataset({"V(out)": ("time", vout)}, coords={"time": time})

    specs = [OvershootSpec(name="os", signal="V(out)", target=1.0)]
    df = measure(ds, specs)
    row = df.row(0, named=True)
    assert row["units"] == "%"
    assert abs(row["value"] - 20.0) < 1e-6
    assert abs(row["peak_time"] - 1.0) < 1e-6


def test_measure_settling_time_abs() -> None:
    time = np.linspace(0.0, 5.0, 6)
    vout = np.array([0.0, 0.5, 0.8, 0.95, 1.02, 1.01])
    ds = xr.Dataset({"V(out)": ("time", vout)}, coords={"time": time})

    specs = [
        SettlingTimeSpec(
            name="settle",
            signal="V(out)",
            target=1.0,
            tolerance=0.05,
            tolerance_kind="abs",
            start_time=1.0,
        )
    ]
    df = measure(ds, specs)
    row = df.row(0, named=True)
    # First index after 1.0 that stays within +/-0.05 is time=3.0
    assert abs(row["value"] - 3.0) < 1e-6
    assert row["units"] == "s"
