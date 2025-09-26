import shutil

import pytest
from spicelab.analysis import NormalPct, monte_carlo
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND
from spicelab.core.types import AnalysisSpec

ng = shutil.which("ngspice")


def _circuit() -> tuple[Circuit, Resistor]:
    c = Circuit("mc_parallel")
    V1 = Vdc("1", 5.0)
    R1 = Resistor("1", 1000.0)
    c.add(V1, R1)
    c.connect(V1.ports[0], R1.ports[0])
    c.connect(R1.ports[1], GND)
    c.connect(V1.ports[1], GND)
    return c, R1


def test_montecarlo_parallel_runs() -> None:
    if not ng:
        pytest.skip("ngspice not installed")

    c, R1 = _circuit()
    mc = monte_carlo(
        c,
        mapping={R1: NormalPct(0.05)},
        n=6,
        analyses=[AnalysisSpec("op", {})],
        engine="ngspice",
        seed=42,
        workers=2,  # paralelo
    )
    assert len(mc.samples) == 6
    assert len(mc.runs) == 6
