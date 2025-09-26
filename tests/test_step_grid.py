import shutil

import pytest
from spicelab.analysis.sweep_grid import run_param_grid
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND
from spicelab.core.types import AnalysisSpec

ng = shutil.which("ngspice")


def _rc_param2() -> tuple[Circuit, Vdc, Resistor, Capacitor]:
    c = Circuit("rc_step2")
    V1 = Vdc("VIN", 1.0)
    R1 = Resistor("R", "1k")
    C1 = Capacitor("C", "100n")
    c.add(V1, R1, C1)
    c.connect(V1.ports[0], R1.ports[0])  # vin
    c.connect(R1.ports[1], C1.ports[0])  # vout
    c.connect(V1.ports[1], GND)
    c.connect(C1.ports[1], GND)
    return c, V1, R1, C1


def test_step_grid_op_runs() -> None:
    if not ng:
        pytest.skip("ngspice not installed")

    c, V1, R1, C1 = _rc_param2()
    result = run_param_grid(
        circuit=c,
        variables=[(V1, [1.0, 5.0]), (R1, ["1k", "2k"]), (C1, ["100n", "220n"])],
        analyses=[AnalysisSpec("op", {})],
        engine="ngspice",
    )
    # 2 * 2 * 2 = 8 combinações
    assert len(result.runs) == 8
    # Deve haver ao menos uma tensão salva
    for run in result.runs:
        ds = run.handle.dataset()
        assert any(var.startswith("V(") for var in ds.data_vars)
