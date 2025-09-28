"""AC measurements demo (synthetic dataset).

Run from repo root:

  uv run --active python examples/ac_measurements_demo.py

Generates a simple integrator-like loop transfer H(jw) = wc/(j*w), then
computes Phase Margin, Gain Bandwidth (unity gain), and Gain Margin.
If matplotlib is available, saves a Bode plot into examples_output/.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from spicelab.analysis import GainBandwidthSpec, GainMarginSpec, PhaseMarginSpec, measure


@dataclass
class _ACDataset:
    freq: np.ndarray
    vout: np.ndarray
    vin: np.ndarray

    @property
    def coords(self) -> dict[str, Any]:
        return {"freq": type("C", (), {"values": self.freq})()}

    @property
    def data_vars(self) -> dict[str, Any]:
        C = self.coords
        return {
            "vout": type("D", (), {"values": self.vout, "dims": ("freq",), "coords": C})(),
            "vin": type("D", (), {"values": self.vin, "dims": ("freq",), "coords": C})(),
        }

    def __getitem__(self, k: str) -> Any:
        return self.data_vars[k]


def build_integrator_dataset(pole_hz: float = 2e3) -> _ACDataset:
    freq = np.logspace(1, 6, 400)  # 10 Hz .. 1 MHz
    w = 2 * np.pi * freq
    wc = 2 * np.pi * pole_hz
    H = wc / (1j * w)  # = 1j^-1 * wc/w
    vin = np.ones_like(H)
    return _ACDataset(freq=freq, vout=H, vin=vin)


def maybe_plot_bode(
    freq: np.ndarray, H: np.ndarray, pm_deg: float, gbw_hz: float, out_dir: Path
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    mag_db = 20 * np.log10(np.abs(H))
    phase_deg = np.unwrap(np.angle(H)) * (180.0 / np.pi)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True)
    ax1.semilogx(freq, mag_db)
    ax1.axhline(0.0, color="k", lw=0.8, ls="--")
    ax1.set_ylabel("|H| [dB]")
    ax1.grid(True, which="both", ls=":", alpha=0.6)
    ax2.semilogx(freq, phase_deg)
    ax2.axhline(-180, color="r", lw=0.8, ls="--", alpha=0.6)
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Phase [deg]")
    ax2.grid(True, which="both", ls=":", alpha=0.6)
    ax1.set_title(f"AC metrics: PM≈{pm_deg:.1f} deg   GBW≈{gbw_hz:.1f} Hz")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ac_measurements_demo_bode.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    ds = build_integrator_dataset(2e3)
    rows = measure(
        ds,
        [
            PhaseMarginSpec(name="pm", numerator="vout", denominator="vin"),
            GainBandwidthSpec(name="gbw", numerator="vout", denominator="vin"),
            GainMarginSpec(name="gm", numerator="vout", denominator="vin", tolerance_deg=20.0),
        ],
        return_as="python",
    )
    rows_by = {r["measure"]: r for r in rows}
    pm = rows_by["pm"]["value"]
    gbw = rows_by["gbw"]["value"]
    gm = rows_by["gm"]["value"]
    print("AC metrics:")
    print(f"  Phase Margin: {pm:.3f} deg")
    print(f"  Gain Bandwidth (unity): {gbw:.3f} Hz")
    print(f"  Gain Margin: {gm if math.isinf(gm) else f'{gm:.3f} dB'}")
    maybe_plot_bode(ds.freq, ds.vout, float(pm), float(gbw), Path("examples_output"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
