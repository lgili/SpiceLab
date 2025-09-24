import shutil

import pytest
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND
from spicelab.spice import ngspice_cli

ng = shutil.which("ngspice")


def test_op_runs_with_ngspice() -> None:
    if not ng:
        pytest.skip("ngspice not installed")

    c = Circuit("op_test")
    V1 = Vdc("1", 5.0)
    R1 = Resistor("1", "1k")
    c.add(V1, R1)
    c.connect(V1.ports[0], R1.ports[0])  # V+
    c.connect(R1.ports[1], GND)
    c.connect(V1.ports[1], GND)
    net = c.build_netlist()
    res = ngspice_cli.run_op(net)
    assert res.returncode == 0
