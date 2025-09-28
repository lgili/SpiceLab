from __future__ import annotations

import numpy as np
from spicelab.analysis.measure import (
    GainSpec,
    OvershootSpec,
    SettlingTimeSpec,
    measure,
)


class _DummyDataset:
    def __init__(self) -> None:
        # Simple AC-like frequency axis and two signals
        self.coords = {"freq": type("C", (), {"values": np.array([1.0, 10.0, 100.0, 1000.0])})()}
        self.data_vars = {
            "vin": type(
                "D",
                (),
                {
                    "values": np.array([1.0, 1.0, 1.0, 1.0]),
                    "dims": ("freq",),
                    "coords": self.coords,
                },
            )(),
            "vout": type(
                "D",
                (),
                {
                    "values": np.array([0.1, 1.0, 10.0, 5.0]),
                    "dims": ("freq",),
                    "coords": self.coords,
                },
            )(),
        }

    def __getitem__(self, key: str):  # xarray-like
        return self.data_vars[key]


def test_measure_gain_db_and_ratio_smoke() -> None:
    ds = _DummyDataset()
    rows = measure(
        ds,
        [
            GainSpec(name="g_db", numerator="vout", denominator="vin", freq=10.0, kind="db"),
            GainSpec(name="g_vv", numerator="vout", denominator="vin", freq=100.0, kind="mag"),
        ],
        return_as="python",
    )
    assert len(rows) == 2
    assert {"measure", "type", "value", "units"}.issubset(rows[0].keys())


def test_measure_overshoot_and_settling_smoke() -> None:
    # Build a step-like signal reaching target=1.0 with small overshoot
    time = np.linspace(0, 1e-3, 100)
    values = 1.0 - np.exp(-time / (0.1e-3))
    values[60] = 1.05  # overshoot ~5%

    class _Sig:
        def __init__(self, t, v):
            self.coords = {"time": type("C", (), {"values": t})()}
            self.values = v
            self.dims = ("time",)

        def __getitem__(self, k):
            return self

    ds = {"vout": _Sig(time, values)}

    class _DS:
        data_vars = ds

        def __getitem__(self, k):
            return ds[k]

    rows = measure(
        _DS(),
        [
            OvershootSpec(name="os", signal="vout", target=1.0, percent=True),
            SettlingTimeSpec(
                name="st",
                signal="vout",
                target=1.0,
                tolerance=0.02,
                tolerance_kind="pct",
                start_time=0.0,
            ),
        ],
        return_as="python",
    )
    assert len(rows) == 2
