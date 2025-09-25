"""RC AC example using the unified engine orchestrator."""

from __future__ import annotations

import argparse
import os

from spicelab.core.circuit import Circuit
from spicelab.core.components import VA, Resistor
from spicelab.core.net import GND
from spicelab.core.types import AnalysisSpec
from spicelab.engines import run_simulation


def build_rc() -> Circuit:
    c = Circuit("rc_ac")
    V1 = VA(ac_mag=1.0)
    R1 = Resistor("1", "1k")
    c.add(V1, R1)
    c.connect(V1.ports[0], R1.ports[0])
    c.connect(V1.ports[1], GND)
    c.connect(R1.ports[1], GND)
    return c


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RC AC sweep via unified engines")
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
    analyses = [AnalysisSpec("ac", {"sweep_type": "dec", "n": 10, "fstart": 10.0, "fstop": 1e6})]
    handle = run_simulation(circuit, analyses, engine=engine)
    ds = handle.dataset()
    print("engine:", handle.attrs().get("engine"))
    print("coords:", list(ds.coords))
    print("vars:", list(ds.data_vars))
