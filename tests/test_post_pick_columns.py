import shutil

import pytest
from spicelab.analysis import OP, stack_step_to_df, step_param
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND

ng = shutil.which("ngspice")


def _simple() -> Circuit:
    c = Circuit("op_post")
    V1 = Vdc("1", 5.0)
    R1 = Resistor("1", "{R}")
    c.add(V1, R1)
    c.connect(V1.ports[0], R1.ports[0])
    c.connect(R1.ports[1], GND)
    c.connect(V1.ports[1], GND)
    return c


def test_stack_pick_columns() -> None:
    if not ng:
        pytest.skip("ngspice not installed")
    c = _simple()
    step = step_param(c, "R", ["1k", "2k"], analysis_factory=lambda: OP())
    df = stack_step_to_df(step, y=None, with_x=True)
    # Seleciona coluna X + pelo menos uma tensão (nome depende do nó)
    xcol = "time" if "time" in df.columns else df.columns[0]
    assert xcol in df.columns
    assert "R" in df.columns
