from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
from _common import savefig
from spicelab.analysis import TRAN, NormalPct, monte_carlo
from spicelab.analysis.viz.plot import plot_mc_metric_hist, plot_traces
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.io.raw_reader import Trace, TraceSet
from spicelab.spice import ngspice_cli


def _rc() -> tuple[Circuit, Resistor, str]:
    c = Circuit("ex_mc_rc")
    V1 = Vdc("1", 5.0)
    R1 = Resistor("1", "1k")
    C1 = Capacitor("1", "1u")
    c.add(V1, R1, C1)
    vin = Net("vin")
    vout = Net("vout")
    c.connect(V1.ports[0], vin)
    c.connect(R1.ports[0], vin)
    c.connect(R1.ports[1], vout)
    c.connect(C1.ports[0], vout)
    c.connect(V1.ports[1], GND)
    c.connect(C1.ports[1], GND)
    return c, R1, "v(vout)"


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

    # 1) Overlay a few transients using the Plotly helper
    indices = _select_indices(len(mc.runs), 6)
    base_ts = mc.runs[0].traces
    time_trace = Trace("time", base_ts["time"].unit, base_ts["time"].values)
    traces = [time_trace]
    for idx in indices:
        ts = mc.runs[idx].traces
        traces.append(Trace(f"run_{idx}", ts[yname].unit, ts[yname].values))
    overlay = TraceSet(traces)
    fig1 = plot_traces(
        overlay,
        title="Monte Carlo — Vout (samples)",
        xlabel="time [s]",
        ylabel="voltage [V]",
    )
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
    fig2 = plot_mc_metric_hist(
        err_pct,
        title=f"Monte Carlo (N={runs}) — error @ {t_sample*1e3:.1f} ms [%]",
        xlabel="Error [%]",
        ylabel="Count",
    )
    savefig(fig2, "mc_hist.png")

    # cleanup artifacts (temporary workdirs)
    for run in mc.runs:
        ngspice_cli.cleanup_artifacts(run.run.artifacts)


if __name__ == "__main__":
    main()
