"""RC transient sweep plotted with Plotly across any engine.

Run from the project root::

    uv run --active python examples/step_sweep_fig.py --engine ngspice
    uv run --active python examples/step_sweep_fig.py --engine ltspice
    uv run --active python examples/step_sweep_fig.py --engine xyce
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:  # pragma: no cover
    from ._common import parser_with_engine, print_header, resolve_engine, savefig
except ImportError:  # pragma: no cover - direct execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from examples._common import (  # type: ignore[import-not-found]
        parser_with_engine,
        print_header,
        resolve_engine,
        savefig,
    )

from spicelab.analysis.sweep_grid import run_param_grid
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.core.types import AnalysisSpec
from spicelab.spice import ngspice_cli
from spicelab.spice.base import RunArtifacts
from spicelab.viz import plot_sweep_df


def build_circuit() -> tuple[Circuit, Resistor, Capacitor]:
    circuit = Circuit("ex_step_grid_tran")
    src = Vdc("VIN", 1.0)
    r_var = Resistor("R", "1k")
    c_var = Capacitor("C", "100n")
    circuit.add(src, r_var, c_var)
    vin = Net("vin")
    vout = Net("vout")
    circuit.connect(src.ports[0], vin)
    circuit.connect(r_var.ports[0], vin)
    circuit.connect(r_var.ports[1], vout)
    circuit.connect(c_var.ports[0], vout)
    circuit.connect(src.ports[1], GND)
    circuit.connect(c_var.ports[1], GND)
    return circuit, r_var, c_var


def _parse_args() -> argparse.Namespace:
    parser = parser_with_engine("RC sweep plotting demo")
    parser.add_argument("--out", default="step_grid.png", help="Image path for the Plotly export")
    parser.add_argument(
        "--keep-artifacts", action="store_true", help="Do not delete engine artefacts after the run"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    engine = resolve_engine(getattr(args, "engine", None))
    print_header("RC sweep (figure)", engine)

    circuit, r_var, c_var = build_circuit()
    grid = {"R": ["1k", "2k", "5k"], "C": ["100n"]}
    result = run_param_grid(
        circuit=circuit,
        variables=[(r_var, grid["R"]), (c_var, grid["C"])],
        analyses=[AnalysisSpec("tran", {"tstep": "50us", "tstop": "5ms"})],
        engine=engine,
    )

    frames: list[pd.DataFrame] = []
    for run in result.runs:
        df = run.handle.dataset().to_dataframe().reset_index()
        for key, value in run.combo.items():
            df[key] = value
        frames.append(df)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        print("No data returned from sweep; nothing to plot.")
        return

    def _find_voltage_column(frame: pd.DataFrame, target: str) -> str | None:
        target_norm = target.lower()
        for col in frame.columns:
            if isinstance(col, str) and col.lower() == target_norm:
                return col
        for col in frame.columns:
            if isinstance(col, str) and col.lower().startswith("v("):
                return col
        return None

    y_column = _find_voltage_column(df, "v(vout)")
    if y_column is None:
        print("Could not locate a voltage column (e.g. 'V(vout)') in the sweep results.")
        print("Available columns:", list(df.columns))
        return

    fig = plot_sweep_df(df, x="time", y=y_column, hue="R", title="RC step (vary R)")
    savefig(fig, args.out)

    if args.keep_artifacts:
        return

    for run in result.runs:
        attrs = run.handle.attrs()
        engine_name = str(attrs.get("engine", engine)).lower()
        if not engine_name.startswith("ngspice"):
            continue
        art = RunArtifacts(
            netlist_path=str(attrs.get("netlist_path")),
            log_path=str(attrs.get("log_path")),
            raw_path=str(attrs.get("raw_path")),
            workdir=str(attrs.get("workdir")),
        )
        ngspice_cli.cleanup_artifacts(art)


if __name__ == "__main__":
    main()
