"""RC transient step response using any supported engine.

Run from the project root:
    uv run --active python examples/rc_tran.py --engine ngspice
    uv run --active python examples/rc_tran.py --engine ltspice
    uv run --active python examples/rc_tran.py --engine xyce

If ``--engine`` is omitted the script falls back to ``ngspice`` or the value of
``SPICELAB_ENGINE``.
"""

from __future__ import annotations

import argparse

try:  # pragma: no cover - executed when running as a script
    from ._common import parser_with_engine, print_header, resolve_engine, run_or_fail
except ImportError:  # pragma: no cover - executed when running as ``python examples/foo.py``
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


def build_circuit() -> Circuit:
    circuit = Circuit("rc_lowpass")
    vin, vout = Net("vin"), Net("vout")
    V1 = Vdc("1", 5.0)
    R1 = Resistor("1", "1k")
    C1 = Capacitor("1", "100n")
    circuit.add(V1, R1, C1)
    circuit.connect(V1.ports[0], vin)
    circuit.connect(V1.ports[1], GND)
    circuit.connect(R1.ports[0], vin)
    circuit.connect(R1.ports[1], vout)
    circuit.connect(C1.ports[0], vout)
    circuit.connect(C1.ports[1], GND)
    return circuit


def _parse_args() -> argparse.Namespace:
    parser = parser_with_engine("RC transient demo")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    engine = resolve_engine(getattr(args, "engine", None))
    print_header("RC transient", engine)
    circuit = build_circuit()
    analyses = [AnalysisSpec("tran", {"tstep": "10us", "tstop": "5ms"})]
    handle = run_or_fail(circuit, analyses, engine=engine)
    ds = handle.dataset()
    print("engine:", handle.attrs().get("engine"))
    print("coords:", list(ds.coords))
    print("vars:", list(ds.data_vars))
