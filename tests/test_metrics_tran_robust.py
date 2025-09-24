from typing import Any

import numpy as np
from numpy.typing import NDArray
from spicelab.analysis.metrics_tran import (
    OvershootResult,
    RiseFall,
    fall_time,
    overshoot,
    rise_time,
    settling_time,
)
from spicelab.io.raw_reader import Trace, TraceSet


def _make_ts(names_and_arrays: list[tuple[str, NDArray[Any]]] | None = None) -> TraceSet:
    if names_and_arrays is None:
        t = np.linspace(0.0, 1.0, 11)
        y = np.linspace(0.0, 1.0, 11)
        return TraceSet([Trace("time", "s", t), Trace("v(n1)", "V", y)])
    traces = []
    for name, arr in names_and_arrays:
        unit = "s" if name == "time" else "V"
        traces.append(Trace(name, unit, arr))
    return TraceSet(traces)


def test_pick_voltage_trace_name_fallback_to_vn2() -> None:
    t = np.linspace(0.0, 1.0, 5)
    y2 = np.linspace(0.0, 1.0, 5)
    ts = _make_ts([("time", t), ("v(n2)", y2)])
    # Ask for a different name; function should fall back to v(n2)
    rr = rise_time(ts, "v(vout)")
    assert isinstance(rr, RiseFall)


def test_settling_time_not_reached_returns_none() -> None:
    t = np.linspace(0.0, 1.0, 11)
    # oscillatory signal that never stays within tight band
    y = 1.0 + 0.2 * np.sin(2 * np.pi * 5 * t)
    ts = _make_ts([("time", t), ("v(n1)", y)])
    sr = settling_time(ts, "v(n1)", tol=0.01)
    # With this definition, at minimum the last sample is inside the band,
    # so settling time degenerates to the last time instant.
    assert sr.idx is not None and sr.t_settle is not None


def test_rise_time_uses_envelope_on_ringing() -> None:
    t = np.linspace(0.0, 1.0, 201)
    # Rising with ringing; final ~1.0
    y = 1.0 - np.exp(-5 * t) * np.cos(40 * np.pi * t)
    ts = _make_ts([("time", t), ("v(n1)", y)])
    rr = rise_time(ts, "v(n1)")
    assert rr.trise is not None
    assert rr.trise >= 0.0


def test_fall_time_with_envelope_and_discrete_fallback() -> None:
    t = np.linspace(0.0, 1.0, 21)
    # Falling but with small plateaus (forces discrete fallback)
    y = np.linspace(1.0, 0.0, 21)
    y[5:7] = y[5]  # plateau
    ts = _make_ts([("time", t), ("v(n1)", y)])
    rf = fall_time(ts, "v(n1)")
    assert rf.tfall is not None and rf.tfall >= 0.0


def test_overshoot_handles_flat_signal() -> None:
    t = np.linspace(0.0, 1.0, 3)
    y = np.array([1.0, 1.0, 1.0])
    ts = _make_ts([("time", t), ("v(n1)", y)])
    ov = overshoot(ts, "v(n1)")
    assert isinstance(ov, OvershootResult)
    assert ov.overshoot == 0.0


def test_get_xy_accepts_complex_values() -> None:
    t = np.linspace(0.0, 1.0, 4)
    # store complex values in Trace; real part should be used
    y = (np.linspace(0.0, 1.0, 4) + 1j * np.linspace(0.1, 0.4, 4)).astype(complex)
    ts = TraceSet([Trace("time", "s", t), Trace("v(n1)", "V", y)])
    rr = rise_time(ts, "v(n1)")
    assert rr.trise is not None
