import shutil

import pytest

from cat.analysis import OP
from cat.core.circuit import Circuit
from cat.core.components import AnalogMux8, Resistor, Vdc
from cat.core.net import GND, Net

ng = shutil.which("ngspice")


@pytest.mark.skipif(not ng, reason="ngspice not installed")
def test_mux_static_sel_op_voltage_divider() -> None:
    """Run OP with AnalogMux8 in static `sel` mode feeding a resistive load.

    Expect V(out_sel) = Vin * RL / (r_series + RL).
    """
    c = Circuit("mux_op")
    vin = Net("vin")
    vout2 = Net("vout2")

    V1 = Vdc("1", 5.0)
    M = AnalogMux8(ref="MU1", r_series=100.0, sel=2, enable_ports=False)
    RL = Resistor("L", 1000.0)

    c.add(V1, M, RL)
    # Connect V1 to mux input
    c.connect(V1.ports[0], vin)
    c.connect(V1.ports[1], GND)
    # Wire mux ports: in->vin, out2->vout2, others to GND to satisfy validation
    # Ports ordering: in, out0..out7
    c.connect(M.ports[0], vin)
    for i in range(8):
        port = M.ports[1 + i]
        if i == 2:
            c.connect(port, vout2)
        else:
            c.connect(port, GND)
    # Load at out2 to ground
    c.connect(RL.ports[0], vout2)
    c.connect(RL.ports[1], GND)

    res = OP().run(c)
    got = float(res.traces["v(vout2)"].values[-1])
    want = 5.0 * 1000.0 / (100.0 + 1000.0)
    assert abs(got - want) <= 1e-6 * max(1.0, abs(want))
