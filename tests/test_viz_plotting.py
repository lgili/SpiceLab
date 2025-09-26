import numpy as np
import pytest
from spicelab.io.raw_reader import Trace, TraceSet
from spicelab.viz import plot_bode, plot_traces


def _ts_real() -> TraceSet:
    t = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 0.5])
    return TraceSet([Trace("time", "s", t), Trace("v(n1)", "V", y)])


def _ts_complex() -> TraceSet:
    f = np.array([1e3, 2e3, 3e3])
    z = np.array([1 + 0j, 0.5 + 0.5j, 0.1 + 0.0j])
    # plot_bode exige que o traço seja complexo; passe valores complexos diretamente
    return TraceSet([Trace("frequency", "Hz", f), Trace("v(out)", "V", z)])


def test_plot_traces_and_bode() -> None:
    plotly = pytest.importorskip("plotly.graph_objects")

    ts_r = _ts_real()
    vf = plot_traces(
        ts_r, ys=["v(n1)"], title="t", xlabel="x", ylabel="y", legend=False, grid=False, tight=False
    )
    assert vf.figure.layout.title.text == "t"
    assert any(trace.name == "v(n1)" for trace in vf.figure.data)

    ts_c = _ts_complex()
    bode = plot_bode(ts_c, "v(out)", unwrap_phase=False, title_mag="m", template="plotly_dark")
    assert isinstance(bode.figure, plotly.Figure)
    assert len(bode.figure.data) == 2
    assert bode.figure.layout.template.layout.paper_bgcolor is not None
