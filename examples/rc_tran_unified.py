"""RC transient example using the unified engine orchestrator."""

from __future__ import annotations

import argparse
import os

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.core.types import AnalysisSpec
from spicelab.engines import run_simulation


def build_rc() -> Circuit:
    c = Circuit("rc_lowpass")
    vin, vout = Net("vin"), Net("vout")
    V1 = Vdc("1", 5.0)
    R1 = Resistor("1", "1k")
    C1 = Capacitor("1", "100n")
    c.add(V1, R1, C1)
    c.connect(V1.ports[0], vin)
    c.connect(V1.ports[1], GND)
    c.connect(R1.ports[0], vin)
    c.connect(R1.ports[1], vout)
    c.connect(C1.ports[0], vout)
    c.connect(C1.ports[1], GND)
    return c


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RC transient via unified engines")
    parser.add_argument(
        "--engine",
        default=os.getenv("SPICELAB_ENGINE", "ngspice"),
        help="Engine name (ngspice, ltspice, xyce, ...). Defaults to SPICELAB_ENGINE or 'ngspice'.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    engine = args.engine
    circuit = build_rc()
    analyses = [AnalysisSpec("tran", {"tstep": 10e-6, "tstop": 5e-3})]
    handle = run_simulation(circuit, analyses, engine=engine)
    ds = handle.dataset()
    print("engine:", handle.attrs().get("engine"))
    print("coords:", list(ds.coords))
    print("vars:", list(ds.data_vars))
