import shutil

import pytest
from cat.analysis import AC
from cat.core.circuit import Circuit
from cat.core.components import VA, Resistor
from cat.core.net import GND


def test_ac_runs_with_vac() -> None:
    if not shutil.which("ngspice"):
        pytest.skip("ngspice not installed")

    # Fonte VAC -> nó vin; R -> terra. Mediremos V(vin) na análise AC.
    c = Circuit("ac_test")
    V1 = VA(ac_mag=1.0)  # 1 V AC
    R1 = Resistor("1", "1k")
    c.add(V1, R1)
    # Conexões: V1.p -> nó vin (compartilhado com R1.a), V1.n -> GND, R1.b -> GND
    c.connect(V1.ports[0], R1.ports[0])  # V1.p == nó vin
    c.connect(V1.ports[1], GND)  # V1.n -> 0
    c.connect(R1.ports[1], GND)  # R1.b -> 0

    res = AC("dec", 10, 10.0, 1e6).run(c)
    # Deve existir o eixo "frequency" e pelo menos mais um traço (tensão de algum nó)
    names = res.traces.names
    assert "frequency" in names or "Frequency" in names  # ngspice usa "frequency"
    assert len(names) >= 2
