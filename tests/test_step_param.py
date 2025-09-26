import shutil

import pytest
from spicelab.analysis.sweep_grid import run_value_sweep
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND
from spicelab.core.types import AnalysisSpec

ng = shutil.which("ngspice")


def _rc_param() -> tuple[Circuit, Resistor]:
    c = Circuit("rc_step")
    V1 = Vdc("1", 5.0)
    R1 = Resistor("1", "1k")
    C1 = Capacitor("1", "100n")
    c.add(V1, R1, C1)
    c.connect(V1.ports[0], R1.ports[0])  # nó vin
    c.connect(R1.ports[1], C1.ports[0])  # nó vout
    c.connect(V1.ports[1], GND)
    c.connect(C1.ports[1], GND)
    return c, R1


def test_step_param_tran_runs() -> None:
    if not ng:
        pytest.skip("ngspice not installed")
    c, R = _rc_param()
    sweep = run_value_sweep(
        c,
        component=R,
        values=["1k", "2k", "5k"],
        analyses=[AnalysisSpec("tran", {"tstep": "100us", "tstop": "2ms"})],
        engine="ngspice",
    )
    assert len(sweep.runs) == 3
    # Cada run deve ter pelo menos eixo "time" + alguma tensão
    for run in sweep.runs:
        ds = run.handle.dataset()
        assert "time" in ds.coords
