import shutil

import pytest
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND
from spicelab.io.raw_reader import parse_ngspice_ascii_raw
from spicelab.spice import ngspice_cli

ng = shutil.which("ngspice")


def test_ascii_forced_tran() -> None:
    if not ng:
        pytest.skip("ngspice not installed")

    c = Circuit("ascii_check")
    V1 = Vdc("1", 5.0)
    R1 = Resistor("1", "1k")
    c.add(V1, R1)
    c.connect(V1.ports[0], R1.ports[0])
    c.connect(R1.ports[1], GND)
    c.connect(V1.ports[1], GND)

    net = c.build_netlist()
    res = ngspice_cli.run_tran(net, tstep="10us", tstop="100us")
    assert res.returncode == 0
    assert res.artifacts.raw_path is not None
    # se for binário, nosso parser ASCII falharia — então isso valida o "set filetype=ascii"
    ts = parse_ngspice_ascii_raw(res.artifacts.raw_path)  # não deve lançar
    assert len(ts.names) >= 2
