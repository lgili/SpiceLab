from __future__ import annotations

import numpy as np
from spicelab.analysis.measure import (
    GainBandwidthSpec,
    PhaseMarginSpec,
    RiseTimeSpec,
    measure,
)


class _TranDataset:
    def __init__(self, t: np.ndarray, y: np.ndarray) -> None:
        self.coords = {"time": type("C", (), {"values": t})()}
        self.data_vars = {
            "y": type("D", (), {"values": y, "dims": ("time",), "coords": self.coords})(),
        }

    def __getitem__(self, k: str):
        return self.data_vars[k]


class _ACDataset:
    def __init__(self, freq: np.ndarray, H: np.ndarray) -> None:
        self.coords = {"freq": type("C", (), {"values": freq})()}
        self.data_vars = {
            "vout": type("D", (), {"values": H, "dims": ("freq",), "coords": self.coords})(),
            "vin": type(
                "D", (), {"values": np.ones_like(H), "dims": ("freq",), "coords": self.coords}
            )(),
        }

    def __getitem__(self, k: str):
        return self.data_vars[k]


def test_rise_time_with_offset_and_no_crossing() -> None:
    # Signal starts at 0.5, ends at 0.6: small change; 10%-90% thresholds around this range.
    t = np.linspace(0, 1e-3, 2001)
    y = 0.5 + 0.1 * np.clip((t - 0.2e-3) / (0.6e-3), 0, 1)
    ds = _TranDataset(t, y)
    # Force thresholds above the final level so that crossing is impossible -> NaN rise time
    rows = measure(
        ds, [RiseTimeSpec(name="tr", signal="y", low_pct=1.05, high_pct=1.10)], return_as="python"
    )
    tr = rows[0]["value"]
    assert not np.isfinite(tr)


def test_ac_interp_fallback_to_nearest_when_no_unity_crossing() -> None:
    # Build a response that never crosses unity (always < 1): H = 0.1 / (1 + j*w/wc)
    freq = np.logspace(1, 5, 64)
    w = 2 * np.pi * freq
    wc = 2 * np.pi * 1e3
    H = 0.1 / (1.0 + 1j * (w / wc))
    ds = _ACDataset(freq, H)
    rows = measure(
        ds,
        [
            PhaseMarginSpec(name="pm", numerator="vout", denominator="vin"),
            GainBandwidthSpec(name="gbw", numerator="vout", denominator="vin"),
        ],
        return_as="python",
    )
    pm = rows[0]["value"]
    gbw = rows[1]["value"]
    # PM should still be finite, derived at nearest point to |H|=1 (here, the maximum magnitude)
    assert np.isfinite(pm)
    # GBW should be a frequency from the sweep (nearest to unity), not NaN
    assert gbw in freq or (freq.min() <= gbw <= freq.max())


def test_rise_time_with_oscillation_multiple_crossings() -> None:
    # Linear ramp 0->1 in 1ms with small oscillation superimposed
    t = np.linspace(0, 1e-3, 5001)
    ramp = np.clip(t / 1e-3, 0, 1)
    osc = 0.05 * np.sin(2 * np.pi * 15e3 * t)  # small ripple
    y = np.clip(ramp + osc, 0, 1)
    ds = _TranDataset(t, y)
    rows = measure(
        ds, [RiseTimeSpec(name="tr", signal="y", low_pct=0.1, high_pct=0.9)], return_as="python"
    )
    tr = rows[0]["value"]
    # Expect near 0.8 ms with some tolerance due to ripple/interpolation
    assert 0.0006 <= tr <= 0.0010


def test_ac_sparse_grid_interpolation_accuracy() -> None:
    # Integrator model: |H| = wc/w, unity at f = wc/(2π) = pole_hz
    pole_hz = 1e3
    freq = np.array([100.0, 300.0, 1000.0, 3000.0, 10000.0])
    w = 2 * np.pi * freq
    wc = 2 * np.pi * pole_hz
    H = wc / (1j * w)
    ds = _ACDataset(freq, H)
    rows = measure(
        ds,
        [
            PhaseMarginSpec(name="pm", numerator="vout", denominator="vin"),
            GainBandwidthSpec(name="gbw", numerator="vout", denominator="vin"),
        ],
        return_as="python",
    )
    rows_by = {r["measure"]: r for r in rows}
    pm = rows_by["pm"]["value"]
    gbw = rows_by["gbw"]["value"]
    # PM should be ~90 deg
    assert 80.0 <= pm <= 100.0
    # GBW close to pole (within 20%) with sparse grid thanks to interpolation in log-f
    assert 0.8 * pole_hz <= gbw <= 1.2 * pole_hz
