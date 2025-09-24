# examples/monte_carlo_rc.py
from __future__ import annotations

import os
import shutil
import sys

import numpy as np
from spicelab.analysis import TRAN, NormalPct, monte_carlo
from spicelab.analysis.viz.plot import plot_mc_metric_hist, plot_traces
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.io.raw_reader import Trace, TraceSet


def _rc_circuit() -> tuple[Circuit, Resistor, str]:
    """
    RC: Vdc -> R -> C -> GND
    Retorna (circuito, componente R para MC, nome do nó de saída).
    """
    c = Circuit("rc_mc_parallel")
    V1 = Vdc("1", 5.0)
    R1 = Resistor("1", "1k")
    C1 = Capacitor("1", "1u")

    c.add(V1, R1, C1)
    vin = Net("vin")
    vout = Net("vout")
    c.connect(V1.ports[0], vin)  # V+ -> R
    c.connect(R1.ports[0], vin)
    c.connect(R1.ports[1], vout)  # R -> C (nó de saída)
    c.connect(C1.ports[0], vout)
    c.connect(V1.ports[1], GND)  # V- -> GND
    c.connect(C1.ports[1], GND)  # C -> GND

    return c, R1, "v(vout)"


def main() -> None:
    if not shutil.which("ngspice"):
        print("ERRO: ngspice não está no PATH. Instale-o antes de rodar este exemplo.")
        sys.exit(1)

    # Monte Carlo — paralelizado via workers da própria lib
    c, R1, yname = _rc_circuit()

    n_samples = 2000  # número de amostras
    sigma = 0.05  # 5% (desvio padrão) no resistor
    t_sample = 1e-3  # medir Vout em 1 ms

    res = monte_carlo(
        c,
        mapping={R1: NormalPct(sigma)},
        n=n_samples,
        analysis_factory=lambda: TRAN("50us", "5ms"),
        seed=42,
        workers=os.cpu_count() or 4,
    )

    # Coleta de Vout(t_sample) em cada execução
    vals: list[float] = []
    for run in res.runs:
        ts = run.traces
        t = ts["time"].values
        y = ts[yname].values
        vals.append(float(np.interp(t_sample, t, y)))

    vals_arr = np.asarray(vals, dtype=float)

    # Referência nominal: pegue a 1ª execução como nominal
    # (ou rode uma simulação "nominal" separada, se preferir)
    v_nom = float(vals_arr[0])
    err_pct = (vals_arr - v_nom) / v_nom * 100.0

    # --- Plots ---

    # 1) Transientes de algumas amostras (sobre o mesmo axes)
    # seleciona 6 índices espaçados ao longo das execuções
    sample_idx = np.linspace(0, len(res.runs) - 1, 6, dtype=int)
    base = res.runs[0].traces
    time_trace = Trace("time", base["time"].unit, base["time"].values)
    traces = [time_trace]
    for idx in sample_idx:
        ts_i = res.runs[idx].traces
        traces.append(Trace(f"run_{idx}", ts_i[yname].unit, ts_i[yname].values))
    overlay = TraceSet(traces)
    fig_overlay = plot_traces(
        overlay,
        title="Monte Carlo (amostras) — Vout",
        xlabel="tempo [s]",
        ylabel="tensão [V]",
    )
    fig_overlay.show()

    # 2) Histograma do erro em %
    fig_hist = plot_mc_metric_hist(
        err_pct,
        title=f"Monte Carlo paralelo (N={n_samples}) — erro @ {t_sample * 1e3:.1f} ms [%]",
        xlabel="Erro [%]",
        ylabel="Contagem",
    )
    fig_hist.show()


if __name__ == "__main__":
    main()
