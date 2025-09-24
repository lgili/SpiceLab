from spicelab.core.circuit import Circuit
from spicelab.core.components import VA, Resistor
from spicelab.core.net import GND
from spicelab.core.types import AnalysisSpec
from spicelab.engines import run_simulation

if __name__ == "__main__":
    # Simple AC analysis demo using the unified orchestrator (ngspice by default)
    c = Circuit("engine_ac_demo")
    V1 = VA(ac_mag=1.0)
    R1 = Resistor("1", "1k")
    c.add(V1, R1)
    c.connect(V1.ports[0], R1.ports[0])
    c.connect(V1.ports[1], GND)
    c.connect(R1.ports[1], GND)

    spec = AnalysisSpec("ac", {"sweep_type": "dec", "n": 10, "fstart": 10.0, "fstop": 1e6})
    handle = run_simulation(c, [spec])

    ds = handle.dataset()
    print(ds)
    print(handle.attrs())
