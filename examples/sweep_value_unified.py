from __future__ import annotations

from spicelab.analysis.sweep_grid import run_value_sweep
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.core.types import AnalysisSpec


def build_rc() -> tuple[Circuit, Resistor]:
    c = Circuit("rc_sweep")
    vin, vout = Net("vin"), Net("vout")
    V1 = Vdc("1", 5.0)
    R1 = Resistor("1", "1k")
    C1 = Capacitor("1", "100n")
    c.add(V1, R1, C1)
    c.connect(V1.ports[0], vin)
    c.connect(V1.ports[1], GND)
    c.connect(R1.ports[0], vin)
    c.connect(R1.ports[1], vout)
    c.connect(C1.ports[0], vout)
    c.connect(C1.ports[1], GND)
    return c, R1


if __name__ == "__main__":
    engine = "ngspice"  # or 'ltspice'/'xyce' if installed
    circuit, R1 = build_rc()
    values = ["1k", "2k", "5k", "10k"]
    analyses = [AnalysisSpec("tran", {"tstep": 10e-6, "tstop": 5e-3})]
    result = run_value_sweep(circuit, R1, values, analyses, engine=engine)
    print("sweep component:", result.component_ref)
    for run in result.runs:
        ds = run.handle.dataset()
        print("value:", run.value, "vars:", list(ds.data_vars)[:3])
