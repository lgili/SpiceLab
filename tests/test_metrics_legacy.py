import numpy as np
from spicelab.analysis import metrics as legacy
from spicelab.io.raw_reader import Trace, TraceSet


def _ts() -> TraceSet:
    t = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 0.5, 0.9, 1.0])
    return TraceSet([Trace("time", "s", t), Trace("v(n1)", "V", y)])


def test_legacy_metrics_cover() -> None:
    ts = _ts()
    _, pk = legacy.peak(ts, "v(n1)")
    st = legacy.settling_time(ts, "v(n1)")
    ov = legacy.overshoot_pct(ts, "v(n1)")
    x, g = legacy.gain_db_from_traces(ts, "v(n1)")
    assert pk >= 1.0 - 1e-12
    assert isinstance(st, float)
    assert ov >= 0.0
    assert len(x) == len(g) == len(ts["time"].values)
