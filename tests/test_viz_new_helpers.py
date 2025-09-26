import numpy as np
import pytest
from spicelab.io.raw_reader import Trace, TraceSet


def _ts_real() -> TraceSet:
    t = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 0.5])
    return TraceSet([Trace("time", "s", t), Trace("v(n1)", "V", y)])


def _ts_complex() -> TraceSet:
    f = np.array([1e3, 2e3, 3e3])
    z = np.array([1 + 0j, 0.5 + 0.5j, 0.1 + 0.0j])
    return TraceSet([Trace("frequency", "Hz", f), Trace("v(out)", "V", z)])


def test_time_series_x_override() -> None:
    pytest.importorskip("plotly.graph_objects")
    from spicelab.viz.plotly import time_series_view

    ts = _ts_real()
    vf = time_series_view(ts, ys=["v(n1)"], x="time", line_width=3, marker_size=6)
    assert any(d.name == "v(n1)" for d in vf.figure.data)


def test_bode_and_nyquist() -> None:
    pytest.importorskip("plotly.graph_objects")
    from spicelab.viz.plotly import bode_view, nyquist_view

    ts = _ts_complex()
    bode = bode_view(ts, "v(out)", x="frequency", unwrap_phase=False)
    assert len(bode.figure.data) == 2
    nyq = nyquist_view(ts, "v(out)")
    assert any(d.name == "v(out)" for d in nyq.figure.data)
