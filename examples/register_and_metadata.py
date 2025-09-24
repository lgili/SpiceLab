"""Example: register a custom component and apply its metadata to a Circuit.

This example shows how to register a component factory that includes metadata
pointing to an external SPICE model file, create an instance of the component,
and have the metadata automatically applied to the circuit using
`apply_metadata_to_circuit`.

Run with:
    uv run --active python register_and_metadata.py
"""

from pathlib import Path
from typing import Any

from spicelab.analysis import OP
from spicelab.core.circuit import Circuit
from spicelab.core.components import Diode, Resistor, Vdc
from spicelab.core.net import GND
from spicelab.library import get_component_spec, register_component, unregister_component
from spicelab.library.utils import apply_metadata_to_circuit
from spicelab.spice import ngspice_cli


def my_resistor_factory(ref: str, *, value: str | float | None = None) -> Resistor:
    # This factory returns a Resistor and we will attach metadata to the
    # registered spec, not to the instance itself.
    return Resistor(ref, value or "1k")


def main(outdir: str | Path | None = None) -> None:
    name = "custom.with_model"
    outdir = Path(outdir) if outdir is not None else Path.cwd()
    models_dir = outdir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "custom_models.lib"
    model_txt = ".model DTEST D(Is=1e-14 N=1)\n.end\n"
    model_path.write_text(model_txt, encoding="utf-8")

    # Register a component with metadata that includes a model file name
    register_component(
        name,
        my_resistor_factory,
        overwrite=True,
        metadata={
            # For this example we include the model file and also provide
            # a model_card entry to demonstrate both behaviors.
            "model_card": [
                ".model Q1 NPN(IS=1e-14 BF=100)",
            ],
            "include": str(model_path),
        },
    )

    try:
        spec = get_component_spec(name)
        circuit = Circuit("with_custom_model")
        added = apply_metadata_to_circuit(circuit, spec)
        print("Directives added to circuit:", added)

        # Build a minimal biased diode circuit: Vdc -> R -> D -> GND
        v1 = Vdc("V1", 1.0)
        r = Resistor("R1", "1k")
        d = Diode("D1", "DTEST")
        circuit.add(v1, r, d)
        circuit.connect(v1.ports[0], r.ports[0])
        circuit.connect(r.ports[1], d.ports[0])
        circuit.connect(d.ports[1], GND)
        circuit.connect(v1.ports[1], GND)

        # Run operating point analysis
        res = OP().run(circuit)
        print("OP run returncode:", res.run.returncode)
        # res.traces may be a strongly typed TraceSet in some runners; cast to
        # Any so static checks don't assume a particular API in examples.
        traces: Any = res.traces
        try:
            keys = list(traces.keys())
        except Exception:
            # Fallback: attempt to iterate over attribute names
            keys = [k for k in getattr(traces, "__dict__", {}).keys()]
        print("OP traces keys:", keys)

        # Cleanup ngspice artifacts created by the run
        ngspice_cli.cleanup_artifacts(res.run.artifacts)
    finally:
        unregister_component(name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir", type=str, default=None, help="output directory for models/artifacts"
    )
    args = parser.parse_args()
    main(outdir=args.outdir)
