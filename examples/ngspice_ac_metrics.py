"""AC metrics (PM/GBW/GM) using ngspice engine on a simple RC.

Run from repo root:

  uv run --active python examples/ngspice_ac_metrics.py

Requires ngspice on PATH. Uses the orchestrator run_simulation helpers indirectly
through the Simulator factory and AnalysisSpec.
"""

from __future__ import annotations

from spicelab.analysis import GainBandwidthSpec, GainMarginSpec, PhaseMarginSpec, measure
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND
from spicelab.core.types import AnalysisSpec
from spicelab.engines.factory import create_simulator


def build_rc() -> Circuit:
    c = Circuit("rc_open_loop")
    vin = Vdc(0.0)  # DC source for node bias; AC is defined in analysis
    r1 = Resistor("1", "1k")
    r2 = Resistor("1", "1k")
    c.add(vin, r1, r2)
    # Tie vin+ to r1, r1->r2->GND, vin- to GND
    c.connect(vin.ports[0], r1.ports[0])
    c.connect(r1.ports[1], r2.ports[0])
    c.connect(r2.ports[1], GND)
    c.connect(vin.ports[1], GND)
    return c


def main() -> int:
    sim = create_simulator("ngspice")
    circuit = build_rc()
    ac = AnalysisSpec("ac", {"sweep_type": "dec", "n": 20, "fstart": 10.0, "fstop": 1e6})
    handle = sim.run(circuit, [ac], None, None)
    ds = handle.dataset()
    # Treat H = V(out)/V(in).
    # Using node names: r1.ports[1] is the mid node; vin.ports[0] is input node name
    # The dataset column naming follows engine conventions like V(node).
    # For tests/examples, we assume:
    num = "v(r1_1)"  # may vary by engine; adjust if needed
    den = "v(r1_0)"
    try:
        rows = measure(
            ds,
            [
                PhaseMarginSpec(name="pm", numerator=num, denominator=den),
                GainBandwidthSpec(name="gbw", numerator=num, denominator=den),
                GainMarginSpec(name="gm", numerator=num, denominator=den, tolerance_deg=20.0),
            ],
            return_as="python",
        )
    except KeyError:
        # Fallback to common node names used by engines: V(r1.1), V(r1.0)
        num = "V(r1.1)"
        den = "V(r1.0)"
        rows = measure(
            ds,
            [
                PhaseMarginSpec(name="pm", numerator=num, denominator=den),
                GainBandwidthSpec(name="gbw", numerator=num, denominator=den),
                GainMarginSpec(name="gm", numerator=num, denominator=den, tolerance_deg=20.0),
            ],
            return_as="python",
        )
    rows_by = {r["measure"]: r for r in rows}
    pm = rows_by["pm"]["value"]
    gbw = rows_by["gbw"]["value"]
    gm = rows_by["gm"]["value"]
    print("ngspice AC metrics (RC divider):")
    print(f"  PM: {pm:.2f} deg  |  GBW: {gbw:.1f} Hz  |  GM: {gm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
