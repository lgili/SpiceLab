from __future__ import annotations

import os
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from _common import savefig

from cat.analysis import TRAN, NormalPct, monte_carlo
from cat.analysis.viz.plot import plot_traces
from cat.core.circuit import Circuit
from cat.core.components import Capacitor, Resistor, Vdc
from cat.core.net import GND
from cat.spice import ngspice_cli


def _rc() -> tuple[Circuit, Resistor, str]:
    c = Circuit("ex_mc_rc")
    V1 = Vdc("1", 5.0)
    R1 = Resistor("1", "1k")
    C1 = Capacitor("1", "1u")
    c.add(V1, R1, C1)
    c.connect(V1.ports[0], R1.ports[0])
    c.connect(R1.ports[1], C1.ports[0])
    c.connect(V1.ports[1], GND)
    c.connect(C1.ports[1], GND)
    return c, R1, "v(n1)"


def _select_indices(n: int, k: int = 6) -> Sequence[int]:
    k = min(k, n)
    return list(np.linspace(0, n - 1, k, dtype=int))


def main() -> None:
    c, R1, yname = _rc()

    # Keep it lightweight for CI docs: ~24 runs
    runs = 24
    t_sample = 1e-3
    mc = monte_carlo(
        circuit=c,
        mapping={R1: NormalPct(0.05)},
        n=runs,
        analysis_factory=lambda: TRAN("50us", "5ms"),
        seed=123,
        workers=min(2, os.cpu_count() or 1),
    )

    # 1) Overlay a few transients
    fig1, ax = plt.subplots()
    ax.set_title("Monte Carlo — Vout (samples)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("voltage [V]")
    for i in _select_indices(len(mc.runs), 6):
        plot_traces(mc.runs[i].traces, ys=[yname], ax=ax, legend=False, grid=True, tight=False)
    fig1.tight_layout()
    savefig(fig1, "mc_traces.png")

    # 2) Histogram of value at t_sample
    vals = []
    for run in mc.runs:
        ts = run.traces
        t = ts["time"].values
        y = ts[yname].values
        vals.append(float(np.interp(t_sample, t, y)))
    vals_arr = np.asarray(vals, dtype=float)
    v_nom = float(vals_arr[0]) if vals_arr.size else 1.0
    err_pct = (vals_arr - v_nom) / max(abs(v_nom), 1e-30) * 100.0
    fig2 = plt.figure()
    plt.hist(err_pct, bins=20, alpha=0.85, edgecolor="black")
    plt.title(f"Monte Carlo (N={runs}) — error @ {t_sample*1e3:.1f} ms [%]")
    plt.xlabel("Error [%]")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    fig2.tight_layout()
    savefig(fig2, "mc_hist.png")

    # cleanup artifacts (temporary workdirs)
    for run in mc.runs:
        ngspice_cli.cleanup_artifacts(run.run.artifacts)


if __name__ == "__main__":
    main()
