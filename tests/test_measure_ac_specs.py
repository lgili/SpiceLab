from __future__ import annotations

import numpy as np
import pytest
from spicelab.analysis.measure import (
    GainBandwidthSpec,
    PhaseMarginSpec,
    measure,
)


class _ACDataset:
    def __init__(self, pole_hz: float = 1e3) -> None:
        # Build a simple single-pole integrator-like response for loop gain:
        # H(jw) = wc / (j*w). This crosses unity at f = pole_hz with -90 deg phase.
        freq = np.logspace(0, 6, 200)  # 1 Hz .. 1 MHz
        w = 2 * np.pi * freq
        wc = 2 * np.pi * pole_hz
        # Avoid w = 0; our sweep starts at 1 Hz so fine.
        H = 1.0 / (1j * (w / wc))  # = wc / (j*w)
        self.coords = {"freq": type("C", (), {"values": freq})()}
        self.data_vars = {
            "vout": type("D", (), {"values": H, "dims": ("freq",), "coords": self.coords})(),
            "vin": type(
                "D", (), {"values": np.ones_like(H), "dims": ("freq",), "coords": self.coords}
            )(),
        }

    def __getitem__(self, k: str):
        return self.data_vars[k]


@pytest.mark.parametrize("pole", [1e3, 2e3])
def test_phase_margin_and_gbw_on_first_order_model(pole: float) -> None:
    ds = _ACDataset(pole_hz=pole)
    rows = measure(
        ds,
        [
            PhaseMarginSpec(name="pm", numerator="vout", denominator="vin"),
            GainBandwidthSpec(name="gbw", numerator="vout", denominator="vin"),
        ],
        return_as="python",
    )
    # PM of a single-pole at unity-gain is ~90 deg
    rows_by_name = {r["measure"]: r for r in rows}
    pm_val = rows_by_name["pm"]["value"]
    assert 60.0 <= pm_val <= 120.0
    # GBW (unity crossover) should be close to the pole frequency
    # (within a factor due to coarse grid)
    gbw_val = rows_by_name["gbw"]["value"]
    assert 0.5 * pole <= gbw_val <= 2.0 * pole
