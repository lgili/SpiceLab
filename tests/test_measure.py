from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest
from spicelab.analysis import GainSpec, OvershootSpec, SettlingTimeSpec, measure
from spicelab.analysis.measure import SignalData, _measure_settling_time
from spicelab.io.raw_reader import Trace, TraceSet

xr = cast(Any, pytest.importorskip("xarray"))


class _FakeFrame:
    def __init__(self, rows: list[dict[str, float | str]]) -> None:
        self._rows = [dict(row) for row in rows]

    @property
    def shape(self) -> tuple[int, int]:
        if not self._rows:
            return (0, 0)
        return (len(self._rows), len(self._rows[0]))

    def row(self, idx: int, named: bool = True) -> dict[str, float | str] | tuple[float | str, ...]:
        row = self._rows[idx]
        return row if named else tuple(row.values())

    @property
    def columns(self) -> list[str]:
        return list(self._rows[0]) if self._rows else []


def _fake_data_frame(rows: list[dict[str, float | str]]) -> _FakeFrame:
    return _FakeFrame(list(rows))


polars_module = cast(Any, ModuleType("polars"))
polars_module.DataFrame = _fake_data_frame
sys.modules["polars"] = polars_module


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


def test_measure_gain_ratio_with_traceset() -> None:
    x = np.array([0.0, 0.5, 1.0])
    vin = np.array([1.0, 1.0, 1.0])
    vout = np.array([0.0, 1.0, 2.0])
    traces = TraceSet(
        [Trace("time", "s", x), Trace("V(in)", None, vin), Trace("V(out)", None, vout)]
    )
    specs = [
        GainSpec(name="gain_mag", numerator="V(out)", denominator="V(in)", freq=1.0, kind="mag")
    ]
    df = measure(traces, specs)
    row = df.row(0, named=True)
    assert row["measure"] == "gain_mag"
    assert row["units"] == "V/V"
    assert abs(row["value"] - 2.0) < 1e-6


def test_measure_gain_infinite_when_denominator_zero() -> None:
    freq = np.array([1.0, 2.0])
    ds = xr.Dataset(
        {
            "V(out)": ("freq", np.array([1.0, 2.0])),
            "V(in)": ("freq", np.array([1.0, 0.0])),
        },
        coords={"freq": freq},
    )
    specs = [GainSpec(name="gain_inf", numerator="V(out)", denominator="V(in)", freq=2.0)]
    row = measure(ds, specs).row(0, named=True)
    assert row["value"] == float("inf")


def test_measure_overshoot_with_reference_lower_target() -> None:
    time = np.array([0.0, 1.0, 2.0])
    vout = np.array([2.0, 1.5, 1.0])
    ds = xr.Dataset({"V(out)": ("time", vout)}, coords={"time": time})
    spec = OvershootSpec(name="os", signal="V(out)", target=1.0, reference=2.0, percent=False)
    row = measure(ds, [spec]).row(0, named=True)
    assert row["units"] == "ratio"
    assert row["value"] == pytest.approx(0.0)


def test_measure_settling_time_pct_and_mismatch_raises() -> None:
    spec = SettlingTimeSpec(name="settle", signal="ignored", target=1.0, tolerance=0.1)
    bad_signal = SignalData(
        axis_name="time", axis=np.array([0.0, 1.0, 2.0]), values=np.array([0.0, 0.5])
    )

    class _StubExtractor:
        def __init__(self, signal: SignalData) -> None:
            self._signal = signal

        def get(self, name: str) -> SignalData:
            return self._signal

    extractor = cast(Any, _StubExtractor(bad_signal))
    with pytest.raises(ValueError):
        _measure_settling_time(extractor, spec)


def test_measure_invalid_signal_raises_value_error() -> None:
    ds = xr.Dataset({}, coords={})
    with pytest.raises(ValueError):
        measure(ds, [GainSpec(name="missing", numerator="V(out)", freq=1.0)])


def test_measure_rejects_unknown_spec() -> None:
    ds = xr.Dataset(
        {"V(out)": ("time", np.array([0.0, 1.0]))}, coords={"time": np.array([0.0, 1.0])}
    )

    class Dummy:
        pass

    with pytest.raises(TypeError):
        measure(ds, [cast(Any, Dummy())])
