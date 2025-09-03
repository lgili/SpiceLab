import shutil

import pytest

from cat.analysis import AC, OP, TRAN
from cat.core.circuit import Circuit
from cat.core.components import Capacitor, Resistor, Vdc
from cat.core.net import GND


def _rc() -> Circuit:
    c = Circuit("rc_api")
    V1 = Vdc("1", 5.0)
    R1 = Resistor("1", "1k")
    C1 = Capacitor("1", "100n")
    c.add(V1, R1, C1)
    c.connect(V1.ports[0], R1.ports[0])  # V+
    c.connect(R1.ports[1], C1.ports[0])  # nó vout
    c.connect(V1.ports[1], GND)
    c.connect(C1.ports[1], GND)
    return c


def test_api_objects_exist() -> None:
    _ = OP()
    _ = TRAN("100us", "1ms")
    _ = AC("dec", 201, 10.0, 1e6)


def test_api_tran_runs() -> None:
    if not shutil.which("ngspice"):
        pytest.skip("ngspice not installed")
    res = TRAN("100us", "2ms").run(_rc())
    assert "time" in res.traces.names
    assert len(res.traces.names) >= 2
