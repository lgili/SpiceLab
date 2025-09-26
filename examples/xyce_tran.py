"""RC transient step response rendered with any supported engine.

Run from the project root (defaults to ngspice)::

    uv run --active python examples/xyce_tran.py --engine ngspice
    uv run --active python examples/xyce_tran.py --engine ltspice
    uv run --active python examples/xyce_tran.py --engine xyce

Optional plot export::

    uv run --active python examples/xyce_tran.py --engine xyce --out-html plots --out-img plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # pragma: no cover
    from ._common import parser_with_engine, print_header, resolve_engine, run_or_fail
except ImportError:  # pragma: no cover - executed directly
    import sys

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

try:  # Plotly helpers are optional
    from spicelab.viz import plot_step_response
except Exception:  # pragma: no cover - optional dependency missing
    plot_step_response = None  # type: ignore[assignment]


def build_circuit() -> Circuit:
    circuit = Circuit("xyce_rc_lowpass")
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
    parser = parser_with_engine("RC transient demo with optional plot export", default="ngspice")
    parser.add_argument(
        "--out-html", default=None, help="Directory to store Plotly HTML (optional)"
    )
    parser.add_argument(
        "--out-img", default=None, help="Directory to store static image (optional)"
    )
    parser.add_argument(
        "--img-format", default="png", help="Image format for static export (png/svg)"
    )
    parser.add_argument(
        "--img-scale", type=float, default=2.0, help="Image scale for static export"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    engine = resolve_engine(getattr(args, "engine", None), fallback="ngspice")
    print_header("RC transient", engine)
    circuit = build_circuit()
    analyses = [AnalysisSpec("tran", {"tstep": "10us", "tstop": "5ms"})]
    try:
        handle = run_or_fail(circuit, analyses, engine=engine)
    except SystemExit:
        print("Ensure the requested engine is installed and configured correctly.")
        return

    ds = handle.dataset()
    attrs = handle.attrs()
    print("engine:", attrs.get("engine", engine))
    print("coords:", list(ds.coords))
    print("vars:", list(ds.data_vars))

    if (args.out_html or args.out_img) and plot_step_response is None:
        print("Plotly not installed; install 'spicelab[viz]' to export plots.")
        return

    if plot_step_response is not None and (args.out_html or args.out_img):
        fig = plot_step_response(ds, "V(vout)")
        if args.out_html:
            out_dir = Path(args.out_html)
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.to_html(out_dir / "xyce_step.html", include_plotlyjs="cdn")
        if args.out_img:
            out_dir = Path(args.out_img)
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                fig.to_image(
                    out_dir / f"xyce_step.{args.img_format}",
                    scale=args.img_scale,
                    format=args.img_format,
                )
            except RuntimeError as exc:  # pragma: no cover - kaleido missing
                print("Image export failed (kaleido missing?):", exc)


if __name__ == "__main__":
    main()
