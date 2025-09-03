# examples/monte_carlo_rc.py
from __future__ import annotations

import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np

from cat import (
    GND,
    TRAN,
    Capacitor,
    Circuit,
    NormalPct,
    Resistor,
    Vdc,
    hist,
    monte_carlo,
    plot_traces,
)


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
    c.connect(V1.ports[0], R1.ports[0])  # V+ -> R
    c.connect(R1.ports[1], C1.ports[0])  # R -> C (nó de saída)
    c.connect(V1.ports[1], GND)  # V- -> GND
    c.connect(C1.ports[1], GND)  # C -> GND

    return c, R1, "v(n1)"


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
        workers=None,  # None => usa todos os núcleos; ajuste se quiser (ex.: workers=4)
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
    fig, ax = plt.subplots()
    ax.set_title("Monte Carlo (amostras) — Vout")
    ax.set_xlabel("tempo [s]")
    ax.set_ylabel("tensão [V]")

    # seleciona 6 índices espaçados ao longo das execuções
    sample_idx = np.linspace(0, len(res.runs) - 1, 6, dtype=int)
    for i in sample_idx:
        ts_i = res.runs[i].traces
        # passa o axes para sobrepor curvas no mesmo gráfico
        plot_traces(ts_i, ys=[yname], ax=ax, legend=True, grid=True, tight=False)

    fig.tight_layout()

    # 2) Histograma do erro em %
    hist(
        err_pct,
        bins=50,
        title=f"Monte Carlo paralelo (N={n_samples}) — erro @ {t_sample * 1e3:.1f} ms [%]",
        xlabel="Erro [%]",
        ylabel="Contagem",
        show_mean=True,
    )

    plt.show()


if __name__ == "__main__":
    main()
