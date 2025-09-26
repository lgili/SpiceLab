import math
import shutil

import pytest
from spicelab.analysis import NormalPct, UniformAbs, monte_carlo
from spicelab.core.circuit import Circuit
from spicelab.core.components import Idc, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.core.types import AnalysisSpec

ng = shutil.which("ngspice")


@pytest.mark.skipif(not ng, reason="ngspice not installed")
def test_mc_identity_res_equals_v_with_1a() -> None:
    """
    Circuit: 1A DC current source into node n1, resistor to GND.
    Ohm's law: V(n1) == R for I=1A. This directly validates that the
    sampled resistor value is injected in the netlist and measured correctly.
    """
    c = Circuit("mc_id_1a")
    n1 = Net("n1")

    I1 = Idc("1", 1.0)  # 1 A from p -> n (flows out of node to GND)
    R1 = Resistor("1", 1000.0)
    c.add(I1, R1)
    c.connect(I1.ports[0], n1)
    c.connect(I1.ports[1], GND)
    c.connect(R1.ports[0], n1)
    c.connect(R1.ports[1], GND)

    mc = monte_carlo(
        c,
        mapping={R1: UniformAbs(50.0)},  # +/- 50 ohms around 1k
        n=12,
        analyses=[AnalysisSpec("op", {})],
        engine="ngspice",
        seed=123,
    )

    assert len(mc.samples) == len(mc.runs) == 12
    for s, run in zip(mc.samples, mc.runs, strict=False):
        r_val = float(s["Resistor.1"])  # sampled value
        v = float(run.traces["V(n1)"].values[-1])
        assert math.isfinite(v)
        # KCL with Idc positive from p->n: V(n1) = -I * R = -1 * R
        assert abs(v + r_val) <= 1e-6 * max(1.0, abs(r_val))


@pytest.mark.skipif(not ng, reason="ngspice not installed")
def test_mc_voltage_divider_matches_formula() -> None:
    """
    Circuit: Vdc -> R1 -> n1 -> R2 -> GND (classic divider).
    Expectation per run: V(n1) == Vin * R2 / (R1 + R2) using the sampled R1/R2.
    """
    c = Circuit("mc_div")
    n1 = Net("n1")
    V1 = Vdc("1", 5.0)
    R1 = Resistor("1", 1000.0)
    R2 = Resistor("2", 2000.0)
    c.add(V1, R1, R2)
    c.connect(V1.ports[0], R1.ports[0])
    c.connect(R1.ports[1], n1)
    c.connect(R2.ports[0], n1)
    c.connect(R2.ports[1], GND)
    c.connect(V1.ports[1], GND)

    mc = monte_carlo(
        c,
        mapping={R1: NormalPct(0.05), R2: NormalPct(0.05)},  # 5% sigma each
        n=16,
        analyses=[AnalysisSpec("op", {})],
        engine="ngspice",
        seed=321,
    )

    assert len(mc.samples) == len(mc.runs) == 16
    vin = 5.0
    for s, run in zip(mc.samples, mc.runs, strict=False):
        r1 = float(s["Resistor.1"])  # sampled values
        r2 = float(s["Resistor.2"])  # sampled values
        want = vin * r2 / max(r1 + r2, 1e-18)
        got = float(run.traces["V(n1)"].values[-1])
        # allow small solver tolerance (~uV level on a few volts)
        assert abs(got - want) <= 1e-6 * max(1.0, abs(vin))
