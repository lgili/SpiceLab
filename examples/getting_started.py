"""Minimal getting-started example for CI/static checks.

This script builds a simple RC circuit and contains guards so it can be
imported during type-checking and linting without requiring plotting or
ngspice binaries.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, cast

import numpy as np
from spicelab.analysis import AC, DC, TRAN, ac_gain_phase
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vac, Vdc
from spicelab.core.net import GND, Net
from spicelab.spice import ngspice_cli

plt: Any | None
try:  # optional plotting dependency
    plt = importlib.import_module("matplotlib.pyplot")
except ModuleNotFoundError:  # pragma: no cover - matplotlib not available
    plt = None

try:
    from examples._common import savefig
except ImportError:

    def savefig(fig: Any, name: str) -> str:  # pragma: no cover - fallback
        fig.savefig(name)
        return name


OUT_DIR = Path("./examples_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_rc_dc() -> Circuit:
    c = Circuit("rc_example")
    v_dc = Vdc("VDC", 1.0)
    r = Resistor("1", "1k")
    cpt = Capacitor("1", "100n")

    vin = Net("vin")
    vout = Net("vout")

    c.add(v_dc, r, cpt)
    c.connect(v_dc.ports[0], vin)
    c.connect(r.ports[0], vin)
    c.connect(r.ports[1], vout)
    c.connect(cpt.ports[0], vout)
    c.connect(v_dc.ports[1], GND)
    c.connect(cpt.ports[1], GND)
    return c


def build_rc_ac() -> Circuit:
    c = Circuit("rc_example_ac")
    v_ac = Vac("VAC", ac_mag=1.0)
    r = Resistor("1", "1k")
    cpt = Capacitor("1", "100n")

    vin = Net("vin")
    vout = Net("vout")

    c.add(v_ac, r, cpt)
    c.connect(v_ac.ports[0], vin)
    c.connect(r.ports[0], vin)
    c.connect(r.ports[1], vout)
    c.connect(cpt.ports[0], vout)
    c.connect(v_ac.ports[1], GND)
    c.connect(cpt.ports[1], GND)
    return c


def run_ac() -> None:
    circuit = build_rc_ac()
    res = AC("dec", 50, 10.0, 1e6).run(circuit)
    freq, mag_db, phase_deg = ac_gain_phase(res.traces, "v(vout)")
    freq_arr = np.asarray(freq, dtype=float)
    mag_arr = np.asarray(mag_db, dtype=float)
    phase_arr = np.asarray(phase_deg, dtype=float)
    if plt is None:  # pragma: no cover
        print("AC samples:", list(zip(freq_arr[:5], mag_arr[:5], strict=False)))
    else:
        fig_ac, (ax_mag, ax_phase) = cast(Any, plt).subplots(2, 1, sharex=True)
        ax_mag.semilogx(freq_arr, mag_arr)
        ax_mag.set_ylabel("Magnitude [dB]")
        ax_mag.grid(True, which="both", alpha=0.3)
        ax_phase.semilogx(freq_arr, phase_arr, color="tab:orange")
        ax_phase.set_ylabel("Phase [deg]")
        ax_phase.set_xlabel("Frequency [Hz]")
        ax_phase.grid(True, which="both", alpha=0.3)
        fig_ac.tight_layout()
        savefig(fig_ac, str(OUT_DIR / "getting_started_ac.png"))
    ngspice_cli.cleanup_artifacts(res.run.artifacts)


def run_dc(c: Circuit) -> None:
    res = DC("VDC", 0.0, 5.0, 0.5).run(c)
    sweep = np.asarray(res.traces["v(v-sweep)"].values, dtype=float)
    vout = np.asarray(res.traces["v(vout)"].values, dtype=float)
    if plt is None:  # pragma: no cover
        print("DC samples:", list(zip(sweep[:5], vout[:5], strict=False)))
    else:
        fig_dc, ax_dc = cast(Any, plt).subplots()
        ax_dc.plot(sweep, vout)
        ax_dc.set_xlabel("Sweep VDC [V]")
        ax_dc.set_ylabel("Vout [V]")
        ax_dc.grid(True, alpha=0.3)
        fig_dc.tight_layout()
        savefig(fig_dc, str(OUT_DIR / "getting_started_dc.png"))
    ngspice_cli.cleanup_artifacts(res.run.artifacts)


def run_tran(c: Circuit) -> None:
    res = TRAN("1e-6", "1e-3", "0.0").run(c)
    if plt is None:  # pragma: no cover
        print("Transient run available")
        return
    time = np.asarray(res.traces["time"].values, dtype=float)
    vout = np.asarray(res.traces["v(vout)"].values, dtype=float)
    fig_tran, ax_tran = cast(Any, plt).subplots()
    ax_tran.plot(time, vout)
    ax_tran.set_xlabel("Time [s]")
    ax_tran.set_ylabel("Vout [V]")
    ax_tran.grid(True, alpha=0.3)
    fig_tran.tight_layout()
    savefig(fig_tran, str(OUT_DIR / "getting_started_tran.png"))
    ngspice_cli.cleanup_artifacts(res.run.artifacts)


def main() -> None:
    run_ac()
    c_dc = build_rc_dc()
    run_dc(c_dc)
    run_tran(c_dc)


if __name__ == "__main__":
    main()
