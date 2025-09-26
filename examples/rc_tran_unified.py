"""RC transient example using the unified engine orchestrator.

Run from the project root:
    uv run --active python examples/rc_tran_unified.py --engine ngspice
    uv run --active python examples/rc_tran_unified.py --engine ltspice
    uv run --active python examples/rc_tran_unified.py --engine xyce
"""

from __future__ import annotations

import argparse

try:  # pragma: no cover
    from ._common import parser_with_engine, print_header, resolve_engine, run_or_fail
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from examples._common import (  # type: ignore[import-not-found]
        parser_with_engine,
        print_header,
        resolve_engine,
        run_or_fail,
    )

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.core.types import AnalysisSpec


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
    return parser_with_engine("RC transient via unified engines").parse_args()


if __name__ == "__main__":
    args = _parse_args()
    engine = resolve_engine(getattr(args, "engine", None))
    print_header("RC transient (unified)", engine)
    circuit = build_rc()
    analyses = [AnalysisSpec("tran", {"tstep": 10e-6, "tstop": 5e-3})]
    handle = run_or_fail(circuit, analyses, engine=engine)
    ds = handle.dataset()
    print("engine:", handle.attrs().get("engine"))
    print("coords:", list(ds.coords))
    print("vars:", list(ds.data_vars))
