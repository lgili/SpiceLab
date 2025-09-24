import shutil

import pytest
from spicelab.analysis import OP, ParamGrid, step_grid
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND

ng = shutil.which("ngspice")


def _rc_param2() -> Circuit:
    c = Circuit("rc_step2")
    V1 = Vdc("1", "{VIN}")
    R1 = Resistor("1", "{R}")
    C1 = Capacitor("1", "{C}")
    c.add(V1, R1, C1)
    c.connect(V1.ports[0], R1.ports[0])  # vin
    c.connect(R1.ports[1], C1.ports[0])  # vout
    c.connect(V1.ports[1], GND)
    c.connect(C1.ports[1], GND)
    return c


def test_step_grid_op_runs() -> None:
    if not ng:
        pytest.skip("ngspice not installed")

    # Tipar explicitamente evita dict[str, object]
    grid: ParamGrid = {
        "VIN": [1.0, 5.0],
        "R": ["1k", "2k"],
        "C": ["100n", "220n"],
    }

    res = step_grid(_rc_param2(), grid, analysis_factory=lambda: OP(), order=["VIN", "R", "C"])
    # 2 * 2 * 2 = 8 combinações
    assert len(res.grid) == 8
    assert len(res.runs) == 8
    # Deve haver ao menos uma tensão salva
    assert all(len(r.traces.names) >= 1 for r in res.runs)
