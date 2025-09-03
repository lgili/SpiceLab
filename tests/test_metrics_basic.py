import numpy as np
from cat.analysis import (
    bandwidth_3db,
    gain_db_from_traces,
    overshoot_pct,
    peak,
    settling_time,
)
from cat.io.raw_reader import Trace, TraceSet


def _ts_step() -> TraceSet:
    # Sinal que sobe de 0->1 com ruído pequeno e estabiliza
    t = np.linspace(0.0, 1e-3, 101)
    y = 1.0 - np.exp(-t / 2e-4)
    # ruído pequeno no final
    y[-10:] = 1.0 + (np.linspace(-1, 1, 10) * 1e-3)
    return TraceSet([Trace("time", "s", t), Trace("v(out)", "V", y)])


def _ts_bode() -> TraceSet:
    # magnitude baixa passada: 1.0 até ~1kHz e cai depois
    f = np.logspace(1, 5, 51)  # 10 .. 1e5
    mag = np.ones_like(f)
    mag[f > 1e3] = 1.0 / np.sqrt(f[f > 1e3] / 1e3)
    return TraceSet([Trace("frequency", "Hz", f), Trace("v(out)", "V", mag)])


def test_peak_and_settling() -> None:
    ts = _ts_step()
    tp, yp = peak(ts, "v(out)")
    assert tp >= 0.0
    st = settling_time(ts, "v(out)", tol=0.02)
    assert 0.0 <= st <= ts.x.values[-1]
    ov = overshoot_pct(ts, "v(out)")
    assert ov >= 0.0


def test_gain_and_bw() -> None:
    ts = _ts_bode()
    x, gdb = gain_db_from_traces(ts, "v(out)")
    assert x.shape == gdb.shape
    bw = bandwidth_3db(ts, "v(out)")
    assert bw is None or (10.0 <= bw <= 1e5)
