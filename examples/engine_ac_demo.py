"""Small-signal AC sweep of an RC divider using any supported engine.

Run from the project root (defaults to ngspice when ``--engine`` is omitted)::

    uv run --active python examples/engine_ac_demo.py --engine ngspice
    uv run --active python examples/engine_ac_demo.py --engine ltspice
    uv run --active python examples/engine_ac_demo.py --engine xyce
"""

from __future__ import annotations

import argparse

try:  # pragma: no cover
    from ._common import parser_with_engine, print_header, resolve_engine, run_or_fail
except ImportError:  # pragma: no cover - script executed directly
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


def build_circuit() -> Circuit:
    circuit = Circuit("engine_ac_demo")
    vin = VA(ac_mag=1.0)
    r_load = Resistor("1", "1k")
    circuit.add(vin, r_load)
    circuit.connect(vin.ports[0], r_load.ports[0])
    circuit.connect(vin.ports[1], GND)
    circuit.connect(r_load.ports[1], GND)
    return circuit


def _parse_args() -> argparse.Namespace:
    return parser_with_engine("AC analysis demo").parse_args()


if __name__ == "__main__":
    args = _parse_args()
    engine = resolve_engine(getattr(args, "engine", None))
    print_header("RC AC sweep", engine)

    circuit = build_circuit()
    analyses = [
        AnalysisSpec(
            "ac",
            {
                "sweep_type": "dec",
                "n": 10,
                "fstart": 10.0,
                "fstop": 1e6,
            },
        )
    ]
    handle = run_or_fail(circuit, analyses, engine=engine)
    ds = handle.dataset()
    print(ds)
    print(handle.attrs())
