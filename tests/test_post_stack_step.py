import shutil

import pytest

from cat.analysis import TRAN, stack_step_to_df, step_param
from cat.core.circuit import Circuit
from cat.core.components import Capacitor, Resistor, Vdc
from cat.core.net import GND

ng = shutil.which("ngspice")


def _rc_param() -> Circuit:
    c = Circuit("rc_post")
    V1 = Vdc("1", 5.0)
    R1 = Resistor("1", "{R}")
    C1 = Capacitor("1", "100n")
    c.add(V1, R1, C1)
    c.connect(V1.ports[0], R1.ports[0])  # vin
    c.connect(R1.ports[1], C1.ports[0])  # vout
    c.connect(V1.ports[1], GND)
    c.connect(C1.ports[1], GND)
    return c


def test_stack_step_tran_df() -> None:
    if not ng:
        pytest.skip("ngspice not installed")
    c = _rc_param()
    step = step_param(c, "R", ["1k", "2k"], analysis_factory=lambda: TRAN("100us", "0.5ms"))
    df = stack_step_to_df(step, y=None, with_x=True)
    # Deve conter a coluna do eixo X (time) e o parâmetro R
    assert "time" in df.columns or "Time" in df.columns
    assert "R" in df.columns
    # Deve haver linhas > 0 e múltiplos runs empilhados
    assert df.shape[0] > 0
    assert df["run_idx"].nunique() == 2
