import numpy as np
from spicelab.analysis.metrics_ac import ac_gain_phase
from spicelab.io.raw_reader import Trace, TraceSet


def test_reconstruct_from_re_im() -> None:
    f = np.array([1.0, 2.0, 3.0])
    re = np.array([1.0, 0.0, -1.0])
    im = np.array([0.0, 1.0, 0.0])
    mag = np.hypot(re, im)
    ts = TraceSet(
        [
            Trace("frequency", None, f),
            Trace("v(out)", None, mag),
            Trace("re(v(out))", None, re),
            Trace("im(v(out))", None, im),
        ]
    )
    f2, mag_db, ph = ac_gain_phase(ts, "v(out)")
    assert f2.shape == mag_db.shape == ph.shape


def test_reconstruct_from_db_and_phase_radians() -> None:
    f = np.array([1.0, 10.0, 100.0])
    db = np.array([0.0, -6.0, -20.0])
    ph_rad = np.array([0.0, -0.5, -1.0])  # radians (|max|<3.5)
    lin = 10 ** (db / 20.0)
    ts = TraceSet(
        [
            Trace("frequency", None, f),
            Trace("v(out)", None, lin),
            Trace("db(v(out))", None, db),
            Trace("ph(v(out))", None, ph_rad),
        ]
    )
    f2, mag_db, ph = ac_gain_phase(ts, "v(out)")
    assert f2.size == 3 and mag_db.size == 3 and ph.size == 3
