from __future__ import annotations

import numpy as np
from spicelab.analysis.measure import ENOBSpec, RiseTimeSpec, THDSpec, measure


class _TranDataset:
    def __init__(self, t: np.ndarray, y: np.ndarray) -> None:
        self.coords = {"time": type("C", (), {"values": t})()}
        self.data_vars = {
            "y": type("D", (), {"values": y, "dims": ("time",), "coords": self.coords})(),
        }

    def __getitem__(self, k: str):
        return self.data_vars[k]


def test_rise_time_basic() -> None:
    t = np.linspace(0, 1e-3, 2001)
    y = np.clip((t - 0.2e-3) / (0.6e-3), 0, 1)
    ds = _TranDataset(t, y)
    rows = measure(
        ds, [RiseTimeSpec(name="tr", signal="y", low_pct=0.1, high_pct=0.9)], return_as="python"
    )
    tr = rows[0]["value"]
    # The slope (0->1) occurs over 0.6ms; 10-90 is 0.8 * 0.6ms = 0.48ms
    assert 0.00045 <= tr <= 0.00051


def test_thd_and_enob_sine() -> None:
    fs = 100e3
    t = np.arange(0, 0.02, 1 / fs)
    f0 = 1e3
    # Fundamental + small 2nd/3rd harmonics
    y = (
        np.sin(2 * np.pi * f0 * t)
        + 0.01 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.005 * np.sin(2 * np.pi * 3 * f0 * t)
    )
    ds = _TranDataset(t, y)
    rows = measure(
        ds,
        [
            THDSpec(name="thd", signal="y", f0=f0, harmonics=5),
            ENOBSpec(name="enob", signal="y", f0=f0, harmonics=5),
        ],
        return_as="python",
    )
    rows_by = {r["measure"]: r for r in rows}
    thd = rows_by["thd"]["value"]
    enob = rows_by["enob"]["value"]
    assert 0.5 <= thd <= 2.5  # ~1% expected
    # With ~1% THD, ENOB should be around ~6 bits
    assert enob > 6.0
