import importlib

import numpy as np
import pytest

from cat.analysis.viz.plot import plot_bode, plot_traces
from cat.io.raw_reader import Trace, TraceSet


def _ts_real() -> TraceSet:
    t = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 0.5])
    return TraceSet([Trace("time", "s", t), Trace("v(n1)", "V", y)])


def _ts_complex() -> TraceSet:
    f = np.array([1e3, 2e3, 3e3])
    z = np.array([1 + 0j, 0.5 + 0.5j, 0.1 + 0.0j])
    # plot_bode exige que o traço seja complexo; passe valores complexos diretamente
    return TraceSet([Trace("frequency", "Hz", f), Trace("v(out)", "V", z)])


@pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None, reason="no matplotlib")
def test_plot_traces_and_bode() -> None:
    ts_r = _ts_real()
    fig = plot_traces(
        ts_r, ys=["v(n1)"], title="t", xlabel="x", ylabel="y", legend=False, grid=False, tight=False
    )
    assert hasattr(fig, "savefig")

    ts_c = _ts_complex()
    f1, f2 = plot_bode(ts_c, "v(out)", unwrap_phase=False, title_mag="m", title_phase="p")
    assert hasattr(f1, "savefig") and hasattr(f2, "savefig")
