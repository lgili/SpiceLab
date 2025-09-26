"""DC operating-point sweep for an RC network using any engine.

Run from the project root::

    uv run --active python examples/step_sweep_grid.py --engine ngspice
    uv run --active python examples/step_sweep_grid.py --engine ltspice
    uv run --active python examples/step_sweep_grid.py --engine xyce
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:  # pragma: no cover
    from ._common import parser_with_engine, print_header, resolve_engine
except ImportError:  # pragma: no cover - direct execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from examples._common import (  # type: ignore[import-not-found]
        parser_with_engine,
        print_header,
        resolve_engine,
    )

from spicelab.analysis.sweep_grid import run_param_grid
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND
from spicelab.core.types import AnalysisSpec


def build_circuit() -> tuple[Circuit, Vdc, Resistor, Capacitor]:
    circuit = Circuit("ex_step_grid")
    src = Vdc("VIN", 1.0)
    r_var = Resistor("R", "1k")
    c_var = Capacitor("C", "100n")
    circuit.add(src, r_var, c_var)
    circuit.connect(src.ports[0], r_var.ports[0])
    circuit.connect(r_var.ports[1], c_var.ports[0])
    circuit.connect(src.ports[1], GND)
    circuit.connect(c_var.ports[1], GND)
    return circuit, src, r_var, c_var


def _parse_args() -> argparse.Namespace:
    parser = parser_with_engine("RC param grid demo")
    parser.add_argument(
        "--out",
        default="step_grid_op.csv",
        help="CSV path to store results (default: step_grid_op.csv)",
    )
    parser.add_argument("--no-save", action="store_true", help="Skip writing the CSV file")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    engine = resolve_engine(getattr(args, "engine", None))
    print_header("RC param grid", engine)

    circuit, src, r_var, c_var = build_circuit()
    grid = {"VIN": [1.0, 5.0], "R": ["1k", "2k"], "C": ["100n", "220n"]}
    result = run_param_grid(
        circuit=circuit,
        variables=[(src, grid["VIN"]), (r_var, grid["R"]), (c_var, grid["C"])],
        analyses=[AnalysisSpec("op", {})],
        engine=engine,
    )

    frames: list[pd.DataFrame] = []
    for run in result.runs:
        df = run.handle.dataset().to_dataframe().reset_index()
        for key, value in run.combo.items():
            df[key] = value
        frames.append(df)

    stacked = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print("columns:", list(stacked.columns))
    if not stacked.empty:
        unique = stacked[["VIN", "R", "C"]].drop_duplicates().shape[0]
        print(f"unique combos: {unique}")
        if not args.no_save:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            stacked.to_csv(out_path, index=False)
            print(f"[saved] {out_path.resolve()}")
