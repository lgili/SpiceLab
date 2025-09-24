import numpy as np
from spicelab.analysis.metrics_ac import (
    ac_gain_phase,
    bandwidth_3db,
    crossover_freq_0db,
    gain_at,
    gain_margin_db,
    loop_gain_bode,
    phase_margin,
)
from spicelab.io.raw_reader import Trace, TraceSet


def _ts_complex_names() -> TraceSet:
    f = np.array([1.0, 2.0, 3.0, 4.0])
    # Provide db and phase explicitly to exercise reconstruction path
    db = np.array([20.0, 10.0, 0.0, -10.0])
    ph = np.array([-10.0, -45.0, -90.0, -135.0])
    mag_lin = 10 ** (db / 20.0)
    return TraceSet(
        [
            Trace("frequency", "Hz", f),
            Trace("v(out)", None, mag_lin),
            Trace("db(v(out))", None, db),
            Trace("phase_deg(v(out))", None, ph),
        ]
    )


def test_ac_gain_phase_and_margins() -> None:
    ts = _ts_complex_names()
    f, mag_db, ph = ac_gain_phase(ts, "v(out)")
    assert f.shape == mag_db.shape == ph.shape
    wc = crossover_freq_0db(ts, "v(out)")
    assert wc is not None
    pm = phase_margin(ts, "v(out)")
    assert pm is not None
    gm = gain_margin_db(ts, "v(out)")
    # phase never reaches -180 in our data; gm may be None
    if gm is not None:
        assert isinstance(gm, float)
    bw = bandwidth_3db(ts, "v(out)")
    assert bw is not None
    g_at = gain_at(ts, "v(out)", f_hz=2.5)
    assert isinstance(g_at, float)


def test_loop_gain_bode_ratio() -> None:
    f = np.array([1.0, 10.0])
    out = np.array([2.0, 1.0])
    inp = np.array([1.0, 1.0])
    ts = TraceSet(
        [Trace("frequency", None, f), Trace("v(out)", None, out), Trace("v(in)", None, inp)]
    )
    f2, mag_db, ph = loop_gain_bode(ts, "v(out)", "v(in)")
    assert f2.shape == mag_db.shape == ph.shape
