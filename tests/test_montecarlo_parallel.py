import shutil

import pytest
from cat.analysis import OP, NormalPct, monte_carlo
from cat.core.circuit import Circuit
from cat.core.components import Resistor, Vdc
from cat.core.net import GND

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
        analysis_factory=lambda: OP(),
        seed=42,
        workers=2,  # paralelo
    )
    assert len(mc.samples) == 6
    assert len(mc.runs) == 6
