"""Plotly visualization demo for CAT's interactive figures.

This script runs a few small analyses and writes Plotly HTML dashboards using
the ``cat.viz`` helpers (``time_series_view``, ``bode_view``, etc.). It requires
the optional ``viz`` extra (``pip install -e '.[viz]'``) so that Plotly and
Kaleido are available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cat.analysis import AC, TRAN, UniformAbs, ac_gain_phase, monte_carlo
from cat.core.circuit import Circuit
from cat.core.components import Capacitor, Resistor, Vac, Vdc
from cat.core.net import GND, Net
from cat.io.raw_reader import Trace, TraceSet
from cat.viz import VizFigure, monte_carlo_histogram, time_series_view
from cat.viz.plotly import _PlotlyNotAvailable


def build_rc_lowpass(
    *,
    ac_source: bool,
    vref: str = "VSRC",
    circuit_name: str = "rc_lowpass",
) -> tuple[Circuit, Resistor]:
    circuit = Circuit(circuit_name)
    resistor = Resistor("R1", "1k")
    capacitor = Capacitor("C1", "100n")
    src = Vac(vref, ac_mag=1.0) if ac_source else Vdc(vref, 1.0)

    vin = Net("vin")
    vout = Net("vout")

    circuit.add(src, resistor, capacitor)
    circuit.connect(src.ports[0], vin)
    circuit.connect(resistor.ports[0], vin)
    circuit.connect(resistor.ports[1], vout)
    circuit.connect(capacitor.ports[0], vout)
    circuit.connect(src.ports[1], GND)
    circuit.connect(capacitor.ports[1], GND)
    return circuit, resistor


def save_figure(fig: VizFigure, out_path: Path, *, include_js: str = "cdn") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.to_html(out_path, include_plotlyjs=include_js)


def produce_bode(outdir: Path) -> None:
    circuit, _ = build_rc_lowpass(ac_source=True, vref="VAC")
    res = AC("dec", 200, 10.0, 1e6).run(circuit)
    freq, mag_db, phase_deg = ac_gain_phase(res.traces, "v(vout)")
    ts = TraceSet(
        [
            Trace("frequency", "Hz", np.asarray(freq, dtype=float)),
            Trace("mag_db", "dB", np.asarray(mag_db, dtype=float)),
            Trace("phase_deg", "deg", np.asarray(phase_deg, dtype=float)),
        ]
    )

    fig_mag = time_series_view(
        ts,
        ys=["mag_db"],
        title="RC Low-pass Magnitude",
        xlabel="Frequency [Hz]",
        ylabel="Magnitude [dB]",
    )
    fig_mag.figure.update_xaxes(type="log")
    save_figure(fig_mag, outdir / "bode_magnitude.html")

    fig_phase = time_series_view(
        ts,
        ys=["phase_deg"],
        title="RC Low-pass Phase",
        xlabel="Frequency [Hz]",
        ylabel="Phase [deg]",
    )
    fig_phase.figure.update_xaxes(type="log")
    save_figure(fig_phase, outdir / "bode_phase.html")


def produce_transient(outdir: Path) -> None:
    circuit, _ = build_rc_lowpass(ac_source=False, vref="VDC")
    res = TRAN("1e-6", "1e-3", "0").run(circuit)
    fig = time_series_view(
        res.traces,
        ys=["v(vout)", "v(vin)"],
        title="Transient response",
        xlabel="Time [s]",
        ylabel="Voltage [V]",
        template="plotly_white",
    )
    save_figure(fig, outdir / "transient.html")


def produce_mc_hist(outdir: Path) -> None:
    circuit, resistor = build_rc_lowpass(ac_source=False, vref="VDC_MC", circuit_name="rc_mc")

    result = monte_carlo(
        circuit,
        {resistor: UniformAbs(0.05)},
        n=30,
        analysis_factory=lambda: TRAN("1e-6", "5e-4", "0"),
        seed=42,
        workers=1,
    )

    final_vout = []
    for run in result.runs:
        trace = run.traces["v(vout)"]
        final_vout.append(float(trace.values[-1]))

    fig = monte_carlo_histogram(
        final_vout,
        title="Monte Carlo: final Vout distribution",
        xlabel="Vout final [V]",
    )
    save_figure(fig, outdir / "monte_carlo_hist.html", include_js="cdn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plotly visualization demo for CAT")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("plotly_output"),
        help="Directory to store generated Plotly HTML files",
    )
    args = parser.parse_args()

    try:
        produce_bode(args.outdir)
        produce_transient(args.outdir)
        produce_mc_hist(args.outdir)
    except _PlotlyNotAvailable as exc:  # pragma: no cover - dependency missing at runtime
        raise SystemExit(
            "Plotly is required for this demo. Install the viz extra via "
            "'pip install -e ' .[viz]''"
        ) from exc


if __name__ == "__main__":
    main()
