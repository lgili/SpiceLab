from __future__ import annotations

import numpy as np
from spicelab.analysis.measure import GainMarginSpec, measure


class _ACDataset:
    def __init__(self, H_values: np.ndarray, freq: np.ndarray) -> None:
        self.coords = {"freq": type("C", (), {"values": freq})()}
        self.data_vars = {
            "vout": type("D", (), {"values": H_values, "dims": ("freq",), "coords": self.coords})(),
            "vin": type(
                "D",
                (),
                {"values": np.ones_like(H_values), "dims": ("freq",), "coords": self.coords},
            )(),
        }

    def __getitem__(self, k: str):
        return self.data_vars[k]


def test_gain_margin_integrator_returns_inf() -> None:
    # Single pole integrator: H = wc / (j*w). Phase is -90° across band; no -180° crossing.
    freq = np.logspace(0, 6, 400)
    w = 2 * np.pi * freq
    wc = 2 * np.pi * 1e3
    H = wc / (1j * w)
    ds = _ACDataset(H_values=H, freq=freq)
    rows = measure(
        ds,
        [GainMarginSpec(name="gm", numerator="vout", denominator="vin", tolerance_deg=10.0)],
        return_as="python",
    )
    gm = rows[0]["value"]
    assert gm == float("inf")


def test_gain_margin_second_order_is_finite() -> None:
    # Second-order plant with two poles: H = wc^2 / (j*w)^2
    # Phase approaches -180° at high frequency, so GM should be finite when crossing threshold.
    freq = np.logspace(0, 6, 400)
    w = 2 * np.pi * freq
    wc = 2 * np.pi * 2e3
    H = (wc**2) / ((1j * w) ** 2)
    ds = _ACDataset(H_values=H, freq=freq)
    rows = measure(
        ds,
        [GainMarginSpec(name="gm", numerator="vout", denominator="vin", tolerance_deg=20.0)],
        return_as="python",
    )
    gm = rows[0]["value"]
    assert np.isfinite(gm)
    # Sanity: magnitude decreases with frequency; GM should be positive dB.
    assert gm > 0.0
