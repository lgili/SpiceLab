import shutil

import pytest
from cat.analysis import TRAN, step_param
from cat.core.circuit import Circuit
from cat.core.components import Capacitor, Resistor, Vdc
from cat.core.net import GND

ng = shutil.which("ngspice")


def _rc_param() -> Circuit:
    c = Circuit("rc_step")
    V1 = Vdc("1", 5.0)
    R1 = Resistor("1", "{R}")
    C1 = Capacitor("1", "100n")
    c.add(V1, R1, C1)
    c.connect(V1.ports[0], R1.ports[0])  # nó vin
    c.connect(R1.ports[1], C1.ports[0])  # nó vout
    c.connect(V1.ports[1], GND)
    c.connect(C1.ports[1], GND)
    return c


def test_step_param_tran_runs() -> None:
    if not ng:
        pytest.skip("ngspice not installed")
    c = _rc_param()
    res = step_param(c, "R", ["1k", "2k", "5k"], analysis_factory=lambda: TRAN("100us", "2ms"))
    assert len(res.grid) == 3
    assert len(res.runs) == 3
    # Cada run deve ter pelo menos eixo "time" + alguma tensão
    assert all("time" in r.traces.names for r in res.runs)
