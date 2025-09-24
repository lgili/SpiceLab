import shutil

import pytest
from spicelab.analysis import OP, sweep_component
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND

ng = shutil.which("ngspice")


def _circuit() -> tuple[Circuit, Resistor]:
    c = Circuit("sweep_py")
    V1 = Vdc("1", 5.0)
    R1 = Resistor("1", "1k")
    c.add(V1, R1)
    c.connect(V1.ports[0], R1.ports[0])
    c.connect(R1.ports[1], GND)
    c.connect(V1.ports[1], GND)
    return c, R1


def test_sweep_component_runs() -> None:
    if not ng:
        pytest.skip("ngspice not installed")

    c, R1 = _circuit()
    sr = sweep_component(c, R1, ["1k", "2k", "5k"], analysis_factory=lambda: OP(), param_name="R")
    assert len(sr.values) == 3
    assert len(sr.runs) == 3
    # todos devem ter pelo menos algum traço além do eixo
    assert all(len(r.traces.names) >= 1 for r in sr.runs)
