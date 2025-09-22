"""Minimal getting-started example for CI/static checks.

This script builds a simple RC circuit and contains guards so it can be
imported during type-checking and linting without requiring plotting or
ngspice binaries.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, cast

from cat.analysis import AC, DC, TRAN, ac_gain_phase
from cat.core.circuit import Circuit
from cat.core.components import VA, Capacitor, Resistor, Vdc
from cat.core.net import GND, Net
from cat.spice import ngspice_cli

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


def build_rc() -> Circuit:
    c = Circuit("rc_example")
    v_ac = VA(ac_mag=1.0)
    v_dc = Vdc("1", 1.0)
    r = Resistor("1", "1k")
    cpt = Capacitor("1", "100n")

    vin = Net("vin")
    vout = Net("vout")

    c.add(v_ac, v_dc, r, cpt)
    c.connect(v_ac.ports[0], vin)
    c.connect(v_dc.ports[0], vin)
    c.connect(r.ports[0], vin)
    c.connect(r.ports[1], vout)
    c.connect(cpt.ports[0], vout)
    c.connect(v_ac.ports[1], GND)
    c.connect(v_dc.ports[1], GND)
    c.connect(cpt.ports[1], GND)
    return c


def run_ac(c: Circuit) -> None:
    res = AC("dec", 50, 10.0, 1e6).run(c)
    f, _, _ = ac_gain_phase(res.traces, "v(vout)")
    if plt is None:  # pragma: no cover
        print("AC samples:", f[:3])
        return
    fig = cast(Any, plt).figure()
    fig.tight_layout()
    savefig(fig, str(OUT_DIR / "getting_started_ac.png"))
    ngspice_cli.cleanup_artifacts(res.run.artifacts)


def run_dc(c: Circuit) -> None:
    res = DC("1", 0.0, 5.0, 0.5).run(c)
    traces_any: Any = res.traces
    try:
        keys = list(traces_any.keys())
    except Exception:
        keys = []
    print("DC run returned traces:", keys)


def run_tran(c: Circuit) -> None:
    res = TRAN("1e-6", "1e-3", "0.0").run(c)
    if plt is None:  # pragma: no cover
        print("Transient run available")
        return
    fig = cast(Any, plt).figure()
    fig.tight_layout()
    savefig(fig, str(OUT_DIR / "getting_started_tran.png"))
    ngspice_cli.cleanup_artifacts(res.run.artifacts)


def main() -> None:
    c = build_rc()
    run_ac(c)
    run_dc(c)
    run_tran(c)


if __name__ == "__main__":
    main()
