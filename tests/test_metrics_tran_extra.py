import numpy as np

from cat.analysis.metrics_tran import fall_time, overshoot, rise_time, settling_time
from cat.io.raw_reader import Trace, TraceSet


def _ts_step_up() -> TraceSet:
    t = np.linspace(0.0, 1.0, 11)
    y = np.array([0.0, 0.1, 0.2, 0.4, 0.7, 0.9, 1.0, 1.02, 1.01, 1.0, 1.0])
    return TraceSet([Trace("time", "s", t), Trace("v(n1)", "V", y)])


def _ts_step_down() -> TraceSet:
    t = np.linspace(0.0, 1.0, 11)
    y = np.array([1.0, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01])
    return TraceSet([Trace("time", "s", t), Trace("v(n1)", "V", y)])


def test_rise_and_settle() -> None:
    ts = _ts_step_up()
    rr = rise_time(ts, "v(n1)")
    st = settling_time(ts, "v(n1)")
    assert rr.trise is None or rr.trise >= 0.0
    assert st.t_settle is None or st.t_settle >= 0.0


def test_fall_and_overshoot() -> None:
    ts = _ts_step_down()
    rf = fall_time(ts, "v(n1)")
    ov = overshoot(ts, "v(n1)")
    assert rf.tfall is None or rf.tfall >= 0.0
    assert ov.overshoot >= 0.0
