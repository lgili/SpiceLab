import math
import shutil

import pytest

from cat.analysis import OP
from cat.core.circuit import Circuit
from cat.core.components import Vdc
from cat.core.net import GND, Net
from cat.utils.topologies import opamp_buffer, opamp_inverting

ng = shutil.which("ngspice")


@pytest.mark.skipif(not ng, reason="ngspice not installed")
def test_opamp_buffer_tracks_input() -> None:
    c = Circuit("oa_buf")
    vin = Net("vin")
    vout = Net("vout")
    V = Vdc("1", 1.0)
    c.add(V)
    c.connect(V.ports[0], vin)
    c.connect(V.ports[1], GND)
    opamp_buffer(c, vin, vout)
    r = OP().run(c)
    got = float(r.traces["v(vout)"].values[-1])
    assert math.isfinite(got)
    assert abs(got - 1.0) <= 2e-6


@pytest.mark.skipif(not ng, reason="ngspice not installed")
def test_opamp_inverting_gain() -> None:
    c = Circuit("oa_inv")
    vin = Net("vin")
    vout = Net("vout")
    V = Vdc("1", 1.0)
    c.add(V)
    c.connect(V.ports[0], vin)
    c.connect(V.ports[1], GND)
    opamp_inverting(c, vin, vout, GND, Rin=1000.0, Rf=2000.0)
    r = OP().run(c)
    got = float(r.traces["v(vout)"].values[-1])
    assert math.isfinite(got)
    assert abs(got - (-2.0)) <= 5e-4
