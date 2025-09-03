import shutil

import pytest

from cat.core.circuit import Circuit
from cat.core.components import Capacitor, Resistor, Vdc
from cat.core.net import GND
from cat.io.raw_reader import parse_ngspice_ascii_raw
from cat.spice import ngspice_cli

ng = shutil.which("ngspice")


def test_tran_runs_and_parses_ascii() -> None:
    if not ng:
        pytest.skip("ngspice not installed")
    # RC de exemplo
    c = Circuit("rc_tran")
    V1 = Vdc("1", 5.0)
    R1 = Resistor("1", "1k")
    C1 = Capacitor("1", "100n")
    c.add(V1, R1, C1)
    c.connect(V1.ports[0], R1.ports[0])  # V+
    c.connect(R1.ports[1], C1.ports[0])  # nó vout
    c.connect(V1.ports[1], GND)
    c.connect(C1.ports[1], GND)
    net = c.build_netlist()
    res = ngspice_cli.run_tran(net, tstep="100us", tstop="5ms")
    assert res.returncode == 0
    assert res.artifacts.raw_path is not None
    ts = parse_ngspice_ascii_raw(res.artifacts.raw_path)
    # Deve ter "time" e tensões
    assert "time" in ts.names
    assert len(ts.names) >= 2
