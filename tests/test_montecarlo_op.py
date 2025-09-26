import shutil

import pytest
from spicelab.analysis import NormalPct, monte_carlo
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND
from spicelab.core.types import AnalysisSpec

ng = shutil.which("ngspice")


def _circuit() -> tuple[Circuit, Resistor]:
    c = Circuit("mc_op")
    V1 = Vdc("1", 5.0)
    R1 = Resistor("1", 1000.0)  # float nominal
    c.add(V1, R1)
    c.connect(V1.ports[0], R1.ports[0])
    c.connect(R1.ports[1], GND)
    c.connect(V1.ports[1], GND)
    return c, R1


def test_montecarlo_op_runs() -> None:
    if not ng:
        pytest.skip("ngspice not installed")

    c, R1 = _circuit()
    res = monte_carlo(
        c,
        mapping={R1: NormalPct(0.05)},  # 5% sigma
        n=8,
        analyses=[AnalysisSpec("op", {})],
        engine="ngspice",
        seed=123,
    )
    assert len(res.samples) == 8
    assert len(res.runs) == 8
    # amostras variaram
    vals = [s["Resistor.1"] for s in res.samples]
    assert min(vals) != max(vals)
