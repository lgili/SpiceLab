import shutil

import pytest
from spicelab.analysis import DC
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND

ng = shutil.which("ngspice")


def test_dc_runs() -> None:
    if not ng:
        pytest.skip("ngspice not installed")

    # V1 -> R1 -> GND
    c = Circuit("dc_test")
    V1 = Vdc("1", 0.0)
    R1 = Resistor("1", "1k")
    c.add(V1, R1)
    c.connect(V1.ports[0], R1.ports[0])
    c.connect(R1.ports[1], GND)
    c.connect(V1.ports[1], GND)

    # Varredura de 0 a 5V em 0.5V
    res = DC("1", 0.0, 5.0, 0.5).run(c)
    # Deve ter eixo (tipicamente "voltage") + pelo menos uma grandeza
    assert len(res.traces.names) >= 2
