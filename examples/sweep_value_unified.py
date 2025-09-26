"""Transient sweep of an RC low-pass across multiple engines.

Run from the project root::

    uv run --active python examples/sweep_value_unified.py --engine ngspice
    uv run --active python examples/sweep_value_unified.py --engine ltspice
    uv run --active python examples/sweep_value_unified.py --engine xyce
"""

from __future__ import annotations

import argparse

try:  # pragma: no cover
    from ._common import parser_with_engine, print_header, resolve_engine
except ImportError:  # pragma: no cover - direct execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from examples._common import (  # type: ignore[import-not-found]
        parser_with_engine,
        print_header,
        resolve_engine,
    )

from spicelab.analysis.sweep_grid import run_value_sweep
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.core.types import AnalysisSpec


def build_rc() -> tuple[Circuit, Resistor]:
    circuit = Circuit("rc_sweep")
    vin, vout = Net("vin"), Net("vout")
    src = Vdc("1", 5.0)
    r_var = Resistor("1", "1k")
    cap = Capacitor("1", "100n")
    circuit.add(src, r_var, cap)
    circuit.connect(src.ports[0], vin)
    circuit.connect(src.ports[1], GND)
    circuit.connect(r_var.ports[0], vin)
    circuit.connect(r_var.ports[1], vout)
    circuit.connect(cap.ports[0], vout)
    circuit.connect(cap.ports[1], GND)
    return circuit, r_var


def _parse_args() -> argparse.Namespace:
    parser = parser_with_engine("RC value sweep demo")
    parser.add_argument(
        "--values",
        nargs="+",
        default=["1k", "2k", "5k", "10k"],
        help="Resistor values to sweep (space-separated list).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    engine = resolve_engine(getattr(args, "engine", None))
    print_header("RC value sweep", engine)

    circuit, resistor = build_rc()
    analyses = [AnalysisSpec("tran", {"tstep": "10us", "tstop": "5ms"})]

    result = run_value_sweep(circuit, resistor, list(args.values), analyses, engine=engine)
    print("component:", result.component_ref)
    for run in result.runs:
        ds = run.handle.dataset()
        vars_preview = list(ds.data_vars)[:3]
        print(f"value={run.value} vars={vars_preview}")
