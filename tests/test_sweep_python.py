import shutil

import pytest
from spicelab.analysis.sweep_grid import run_value_sweep
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND
from spicelab.core.types import AnalysisSpec

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
    sr = run_value_sweep(
        circuit=c,
        component=R1,
        values=["1k", "2k", "5k"],
        analyses=[AnalysisSpec("op", {})],
        engine="ngspice",
    )
    assert len(sr.runs) == 3
    # todos devem ter pelo menos algum traço além do eixo
    for run in sr.runs:
        ds = run.handle.dataset()
        assert any(name.startswith("V(") for name in ds.data_vars)
