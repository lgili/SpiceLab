import numpy as np

from cat.analysis.metrics_tran import overshoot_pct
from cat.io.raw_reader import Trace, TraceSet


def test_overshoot_pct_simple_step() -> None:
    # Step 0 -> 1 with peak at 1.2, expect 20% overshoot.
    t = np.linspace(0.0, 1.0, 6)
    y = np.array([0.0, 0.2, 0.6, 1.2, 1.05, 1.0])
    ts = TraceSet([Trace("time", "s", t), Trace("v(n1)", "V", y)])
    pct = overshoot_pct(ts, "v(n1)")
    assert abs(pct - 20.0) <= 1e-6
