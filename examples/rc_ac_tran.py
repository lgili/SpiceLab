"""Run RC transient and AC analyses with any supported SPICE engine.

Run from the project root:
    uv run --active python examples/rc_ac_tran.py --engine ngspice
    uv run --active python examples/rc_ac_tran.py --engine ltspice
    uv run --active python examples/rc_ac_tran.py --engine xyce

Use ``--out-html`` / ``--out-img`` to export Plotly figures. When Plotly is not
installed the script prints a short summary instead of rendering charts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # pragma: no cover - runtime import guard for direct execution
    from ._common import (
        parser_with_engine,
        print_header,
        resolve_engine,
        run_or_fail,
    )
except ImportError:  # pragma: no cover - running as ``python examples/foo.py``
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from examples._common import (  # type: ignore[import-not-found]
        parser_with_engine,
        print_header,
        resolve_engine,
        run_or_fail,
    )

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vac, Vpulse
from spicelab.core.net import GND, Net
from spicelab.core.types import AnalysisSpec
from spicelab.io.raw_reader import TraceSet

try:  # Plotly helpers are optional
    from spicelab.viz import plot_bode, plot_nyquist, plot_traces
except Exception:  # pragma: no cover - Plotly not installed
    plot_bode = plot_nyquist = plot_traces = None  # type: ignore[assignment]


def build_tran_circuit() -> Circuit:
    circuit = Circuit("rc_tran")
    vin, vout = Net("vin"), Net("vout")
    src = Vpulse("STEP", 0.0, 5.0, "0", "1u", "1u", "2ms", "4ms")
    r1 = Resistor("R1", "1k")
    c1 = Capacitor("C1", "100n")
    circuit.add(src, r1, c1)
    circuit.connect(src.ports[0], vin)
    circuit.connect(src.ports[1], GND)
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], vout)
    circuit.connect(c1.ports[0], vout)
    circuit.connect(c1.ports[1], GND)
    return circuit


def build_ac_circuit() -> Circuit:
    circuit = Circuit("rc_ac")
    vin, vout = Net("vin"), Net("vout")
    src = Vac("AC", ac_mag=1.0)
    r1 = Resistor("R1", "1k")
    c1 = Capacitor("C1", "100n")
    circuit.add(src, r1, c1)
    circuit.connect(src.ports[0], vin)
    circuit.connect(src.ports[1], GND)
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], vout)
    circuit.connect(c1.ports[0], vout)
    circuit.connect(c1.ports[1], GND)
    return circuit


def _parser() -> argparse.ArgumentParser:
    parser = parser_with_engine("RC AC + TRAN demo")
    parser.add_argument("--out-html", default=None, help="Directory to store Plotly HTML figures")
    parser.add_argument("--out-img", default=None, help="Directory to store static images")
    parser.add_argument("--img-format", default="png", help="Static image format (png/svg)")
    parser.add_argument("--img-scale", type=float, default=2.0, help="Static image scale factor")
    parser.add_argument(
        "--no-show", action="store_true", help="Do not open interactive Plotly windows"
    )
    return parser


def _export_figure(fig, stem: str, args) -> None:
    if fig is None:
        return
    if args.out_html:
        out_dir = Path(args.out_html)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.to_html(out_dir / f"{stem}.html", include_plotlyjs="cdn")
    if args.out_img:
        out_dir = Path(args.out_img)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            fig.to_image(
                out_dir / f"{stem}.{args.img_format}", scale=args.img_scale, format=args.img_format
            )
        except RuntimeError as exc:  # pragma: no cover - kaleido missing
            print(f"Image export failed ({exc}). Install 'kaleido' for static exports.")


def main() -> None:
    args = _parser().parse_args()
    engine = resolve_engine(getattr(args, "engine", None))
    print_header("RC AC/TRAN", engine)

    tran_handle = run_or_fail(
        build_tran_circuit(),
        [AnalysisSpec("tran", {"tstep": "10us", "tstop": "5ms"})],
        engine=engine,
    )
    tran_ds = tran_handle.dataset()
    print("Transient variables:", list(tran_ds.data_vars))

    ac_handle = run_or_fail(
        build_ac_circuit(),
        [AnalysisSpec("ac", {"sweep_type": "dec", "n": 40, "fstart": 10.0, "fstop": 1e6})],
        engine=engine,
    )
    ac_ds = ac_handle.dataset()
    print("AC variables:", list(ac_ds.data_vars))

    if plot_traces is None or plot_bode is None or plot_nyquist is None:
        print("Plotly not installed; skipping interactive figures.")
        return

    tran_ts = TraceSet.from_dataset(tran_ds)
    ac_ts = TraceSet.from_dataset(ac_ds)

    tran_fig = plot_traces(
        tran_ts, ys=["V(vin)", "V(vout)"], title="RC transient response", x="time"
    )
    bode_fig = plot_bode(ac_ts, "V(vout)", title_mag="RC magnitude response")
    nyq_fig = plot_nyquist(ac_ts, "V(vout)", title="RC Nyquist plot")

    _export_figure(tran_fig, "rc_tran", args)
    _export_figure(bode_fig, "rc_bode", args)
    _export_figure(nyq_fig, "rc_nyquist", args)

    if not args.no_show:
        tran_fig.show()
        bode_fig.show()
        nyq_fig.show()


if __name__ == "__main__":
    main()
