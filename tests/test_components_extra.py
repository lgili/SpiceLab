import math
import shutil

import numpy as np
import pytest
from spicelab.analysis import AC, OP, TRAN
from spicelab.core.circuit import Circuit
from spicelab.core.components import (
    VCCS,
    VCVS,
    Capacitor,
    Diode,
    Iac,
    Ipulse,
    Resistor,
    Vac,
    Vdc,
)
from spicelab.core.net import GND, Net

ng = shutil.which("ngspice")


@pytest.mark.skipif(not ng, reason="ngspice not installed")
def test_ac_current_source_into_resistor() -> None:
    """AC with 1 A current source into R to ground: V = I*R => 1A * 1k = 1k V.

    Checks Iac card formatting and AC magnitude parsing.
    """
    c = Circuit("ac_i_into_r")
    n1 = Net("n1")
    I1 = Iac("1", ac_mag=1.0)
    R1 = Resistor("1", 1000.0)
    c.add(I1, R1)
    c.connect(I1.ports[0], n1)
    c.connect(I1.ports[1], GND)
    c.connect(R1.ports[0], n1)
    c.connect(R1.ports[1], GND)

    res = AC("lin", 1, 1.0, 1.0).run(c)
    mag = float(res.traces["v(n1)"].values[-1])
    assert abs(mag - 1000.0) <= 1e-6


@pytest.mark.skipif(not ng, reason="ngspice not installed")
def test_tran_current_pulse_charging_cap() -> None:
    """Current pulse into capacitor: dv/dt = I/C. For I=1mA, C=1uF, dv/dt=1000V/s.

    After 1 ms, expect ~1 V.
    """
    c = Circuit("tran_i_pulse_cap")
    n1 = Net("n1")
    Ip = Ipulse("1", 0.0, 1e-3, 0.0, 1e-6, 1e-6, 10e-3, 20e-3)
    C1 = Capacitor("1", 1e-6)
    # Small bleed resistor to ground for numerical stability
    Rb = Resistor("b", 1e9)
    c.add(Ip, C1, Rb)
    # Orient source so positive current charges the cap (GND -> n1)
    c.connect(Ip.ports[0], GND)
    c.connect(Ip.ports[1], n1)
    c.connect(C1.ports[0], n1)
    c.connect(C1.ports[1], GND)
    c.connect(Rb.ports[0], n1)
    c.connect(Rb.ports[1], GND)

    res = TRAN("0.1ms", "2ms").run(c)
    ts = res.traces
    t = ts[ts.x.name].values
    v = ts["v(n1)"].values
    # sample near 1ms
    idx = int(np.searchsorted(t, 1e-3))
    got = float(v[min(idx, len(v) - 1)])
    assert math.isfinite(got)
    assert abs(got - 1.0) <= 5e-2


@pytest.mark.skipif(not ng, reason="ngspice not installed")
def test_vcvs_and_vccs_basic_op() -> None:
    """Independent checks for VCVS and VCCS cards under OP.

    - VCVS: gain=2, Vin=1V -> Vout=2V with load.
    - VCCS: gm=1mS, Vin=1V -> I=1mA -> Vout=1V on 1k load.
    """
    # VCVS
    c1 = Circuit("vcvs")
    vin = Net("vin")
    vout = Net("vout")
    V1 = Vdc("1", 1.0)
    E1 = VCVS("1", 2.0)
    RL = Resistor("L", 1000.0)
    c1.add(V1, E1, RL)
    c1.connect(V1.ports[0], vin)
    c1.connect(V1.ports[1], GND)
    c1.connect(E1.ports[2], vin)  # cp
    c1.connect(E1.ports[3], GND)  # cn
    c1.connect(E1.ports[0], vout)  # p (out)
    c1.connect(E1.ports[1], GND)  # n (out return)
    c1.connect(RL.ports[0], vout)
    c1.connect(RL.ports[1], GND)
    r1 = OP().run(c1)
    v_vcvs = float(r1.traces["v(vout)"].values[-1])
    assert abs(v_vcvs - 2.0) <= 1e-6

    # VCCS
    c2 = Circuit("vccs")
    vin2 = Net("vin")
    vout2 = Net("vout")
    V2 = Vdc("2", 1.0)
    G1 = VCCS("1", 1e-3)  # 1 mS
    RL2 = Resistor("L", 1000.0)
    c2.add(V2, G1, RL2)
    c2.connect(V2.ports[0], vin2)
    c2.connect(V2.ports[1], GND)
    c2.connect(G1.ports[2], vin2)  # cp
    c2.connect(G1.ports[3], GND)  # cn
    # Orient VCCS so positive current raises vout (current GND -> vout)
    c2.connect(G1.ports[0], GND)  # p
    c2.connect(G1.ports[1], vout2)  # n
    c2.connect(RL2.ports[0], vout2)
    c2.connect(RL2.ports[1], GND)
    r2 = OP().run(c2)
    v_vccs = float(r2.traces["v(vout)"].values[-1])
    assert abs(v_vccs - 1.0) <= 1e-6


@pytest.mark.skipif(not ng, reason="ngspice not installed")
def test_diode_forward_conduction_bounds() -> None:
    """Simple DC bias of diode through resistor — voltage across diode ~0.4..0.9 V.

    Ensures .model directive handling and basic nonlinear solve.
    """
    c = Circuit("d_fwd")
    n1 = Net("n1")
    V = Vdc("1", 1.0)
    R = Resistor("1", 1000.0)
    D = Diode("1", "D1")
    c.add_directive(".model D1 D(Is=1e-14)")
    c.add(V, R, D)
    c.connect(V.ports[0], R.ports[0])
    c.connect(R.ports[1], n1)
    c.connect(D.ports[0], n1)
    c.connect(D.ports[1], GND)
    c.connect(V.ports[1], GND)

    res = OP().run(c)
    vd = float(res.traces["v(n1)"].values[-1])
    assert 0.4 <= vd <= 0.9


@pytest.mark.skipif(not ng, reason="ngspice not installed")
def test_ac_lowpass_monotonic_mag() -> None:
    """Magnitude of RC low-pass decreases from low to high frequency (10 Hz -> 10 kHz)."""
    c = Circuit("ac_lp_rc")
    vin = Net("vin")
    vout = Net("vout")
    V1 = Vac("1", ac_mag=1.0)
    R1 = Resistor("1", 1000.0)
    C1 = Capacitor("1", 1e-6)
    c.add(V1, R1, C1)
    c.connect(V1.ports[0], vin)
    c.connect(V1.ports[1], GND)
    c.connect(R1.ports[0], vin)
    c.connect(R1.ports[1], vout)
    c.connect(C1.ports[0], vout)
    c.connect(C1.ports[1], GND)

    res = AC("dec", 10, 10.0, 1e4).run(c)
    mags = res.traces["v(vout)"].values
    assert mags[0] >= mags[-1]
