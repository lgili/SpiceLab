"""Transient measurements demo (synthetic signals).

Run from repo root:

  uv run --active python examples/tran_measurements_demo.py

Generates a ramped step for rise time, and a sine with weak harmonics
for THD/ENOB estimation. If matplotlib is available, saves plots into
examples_output/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from spicelab.analysis import ENOBSpec, RiseTimeSpec, THDSpec, measure


@dataclass
class _TranDataset:
    time: np.ndarray
    y: np.ndarray

    @property
    def coords(self) -> dict[str, Any]:
        return {"time": type("C", (), {"values": self.time})()}

    @property
    def data_vars(self) -> dict[str, Any]:
        C = self.coords
        return {
            "y": type("D", (), {"values": self.y, "dims": ("time",), "coords": C})(),
        }

    def __getitem__(self, k: str) -> Any:
        return self.data_vars[k]


def build_ramp_step() -> _TranDataset:
    t = np.linspace(0, 1e-3, 2001)
    y = np.clip((t - 0.2e-3) / (0.6e-3), 0, 1)
    return _TranDataset(time=t, y=y)


def build_sine_with_harmonics(fs: float = 100e3, f0: float = 1e3) -> _TranDataset:
    t = np.arange(0, 0.02, 1 / fs)
    y = (
        np.sin(2 * np.pi * f0 * t)
        + 0.01 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.005 * np.sin(2 * np.pi * 3 * f0 * t)
    )
    return _TranDataset(time=t, y=y)


def maybe_plot_tran(t: np.ndarray, y: np.ndarray, title: str, name: str, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.0))
    ax.plot(t, y)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("y")
    ax.grid(True, ls=":", alpha=0.6)
    ax.set_title(title)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    out_dir = Path("examples_output")
    # Rise time demo
    step = build_ramp_step()
    rows = measure(step, [RiseTimeSpec(name="tr", signal="y")], return_as="python")
    tr = rows[0]["value"]
    print(f"Rise time (10-90): {tr:.6e} s")
    maybe_plot_tran(step.time, step.y, f"Rise time ≈ {tr * 1e6:.2f} µs", "tran_rise_time", out_dir)

    # THD/ENOB demo
    sine = build_sine_with_harmonics()
    rows2 = measure(
        sine,
        [THDSpec(name="thd", signal="y", f0=1e3), ENOBSpec(name="enob", signal="y", f0=1e3)],
        return_as="python",
    )
    thd = rows2[0]["value"]
    enob = rows2[1]["value"]
    print(f"THD: {thd:.3f} %   ENOB: {enob:.2f} bits")
    maybe_plot_tran(sine.time, sine.y, "Sine with small harmonics", "tran_sine_harmonics", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
