import math
import shutil

import numpy as np
import pytest

from cat.analysis import AC, DC, OP, TRAN
from cat.core.circuit import Circuit
from cat.core.components import Capacitor, OpAmpIdeal, Resistor, Vac, Vdc, Vpulse
from cat.core.net import GND, Net

ng = shutil.which("ngspice")


@pytest.mark.skipif(not ng, reason="ngspice not installed")
def test_op_with_ideal_oa_follower() -> None:
    """Unity-gain follower with ideal op-amp tracks the input at DC."""
    c = Circuit("op_oa_follow")
    vin = Net("vin")
    vout = Net("vout")
    V1 = Vdc("1", 1.234)
    oa = OpAmpIdeal("1", gain=1e6)
    c.add(V1, oa)
    c.connect(V1.ports[0], vin)
    c.connect(V1.ports[1], GND)
    # non-inverting input to vin; output to vout; negative input tied to vout
    c.connect(oa.ports[0], vin)
    c.connect(oa.ports[2], vout)
    c.connect(oa.ports[1], vout)

    res = OP().run(c)
    got = float(res.traces["v(vout)"].values[-1])
    assert math.isfinite(got)
    # allow small numerical error due to finite gain and solver tolerances
    assert abs(got - 1.234) <= 2e-6


@pytest.mark.skipif(not ng, reason="ngspice not installed")
def test_ac_divider_single_point() -> None:
    """AC divider magnitude equals R2/(R1+R2) at any frequency for purely resistive net."""
    c = Circuit("ac_div")
    vin = Net("vin")
    vout = Net("vout")
    V1 = Vac("1", ac_mag=1.0)
    R1 = Resistor("1", 1000.0)
    R2 = Resistor("2", 2000.0)
    c.add(V1, R1, R2)
    c.connect(V1.ports[0], vin)
    c.connect(V1.ports[1], GND)
    c.connect(R1.ports[0], vin)
    c.connect(R1.ports[1], vout)
    c.connect(R2.ports[0], vout)
    c.connect(R2.ports[1], GND)

    res = AC("lin", 1, 1.0, 1.0).run(c)
    mag = float(res.traces["v(vout)"].values[-1])
    want = 2000.0 / (1000.0 + 2000.0)
    assert abs(mag - want) <= 1e-6


@pytest.mark.skipif(not ng, reason="ngspice not installed")
def test_tran_rc_step_reaches_final_value() -> None:
    """RC low-pass with step input reaches ~1V after several time constants."""
    c = Circuit("tran_rc")
    vin = Net("vin")
    vout = Net("vout")
    Vp = Vpulse("1", 0.0, 1.0, 0.0, 1e-6, 1e-6, 1.0, 2.0)
    R1 = Resistor("1", 1000.0)
    C1 = Capacitor("1", 1e-6)
    c.add(Vp, R1, C1)
    c.connect(Vp.ports[0], vin)
    c.connect(Vp.ports[1], GND)
    c.connect(R1.ports[0], vin)
    c.connect(R1.ports[1], vout)
    c.connect(C1.ports[0], vout)
    c.connect(C1.ports[1], GND)

    res = TRAN("0.1ms", "6ms").run(c)
    ts = res.traces
    t = ts[ts.x.name].values
    y = ts["v(vout)"].values
    # last sample should be near 1V; allow small ripple due to finite pulse edges
    assert y[-1] >= 0.99
    # monotonic after 1ms roughly
    idx = np.searchsorted(t, 1e-3)
    diffs = np.diff(y[idx:])
    assert np.all(diffs >= -1e-3)


@pytest.mark.skipif(not ng, reason="ngspice not installed")
def test_dc_sweep_voltage_divider_endpoints() -> None:
    """DC sweep of source in a divider hits expected endpoints at 0V and 5V."""
    c = Circuit("dc_div")
    n1 = Net("n1")
    V1 = Vdc("1", 0.0)
    R1 = Resistor("1", 1000.0)
    R2 = Resistor("2", 2000.0)
    c.add(V1, R1, R2)
    c.connect(V1.ports[0], R1.ports[0])
    c.connect(R1.ports[1], n1)
    c.connect(R2.ports[0], n1)
    c.connect(R2.ports[1], GND)
    c.connect(V1.ports[1], GND)

    res = DC("1", 0.0, 5.0, 1.0).run(c)
    y = res.traces["v(n1)"].values
    # Expected endpoints: 0V and 5*R2/(R1+R2) = 10/3 V
    assert abs(float(y[0]) - 0.0) <= 5e-4
    assert abs(float(y[-1]) - (5.0 * 2000.0 / 3000.0)) <= 5e-3
