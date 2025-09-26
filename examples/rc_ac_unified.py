"""Small-signal AC sweep of an RC low-pass across any engine.

Run from the project root (defaults to ngspice)::

    uv run --active python examples/rc_ac_unified.py --engine ngspice
    uv run --active python examples/rc_ac_unified.py --engine ltspice
    uv run --active python examples/rc_ac_unified.py --engine xyce
"""

from __future__ import annotations

import argparse

try:  # pragma: no cover
    from ._common import parser_with_engine, print_header, resolve_engine, run_or_fail
except ImportError:  # pragma: no cover - direct execution support
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
from spicelab.core.components import VA, Resistor
from spicelab.core.net import GND
from spicelab.core.types import AnalysisSpec


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
    return parser_with_engine("RC AC sweep via unified engines").parse_args()


if __name__ == "__main__":
    args = _parse_args()
    engine = resolve_engine(getattr(args, "engine", None))
    print_header("RC AC sweep", engine)
    circuit = build_rc()
    analyses = [AnalysisSpec("ac", {"sweep_type": "dec", "n": 10, "fstart": 10.0, "fstop": 1e6})]
    handle = run_or_fail(circuit, analyses, engine=engine)
    ds = handle.dataset()
    print("engine:", handle.attrs().get("engine"))
    print("coords:", list(ds.coords))
    print("vars:", list(ds.data_vars))
