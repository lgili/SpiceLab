import shutil

import numpy as np
import pytest
from spicelab.analysis import NormalPct, monte_carlo
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.core.types import AnalysisSpec

ng = shutil.which("ngspice")


@pytest.mark.skipif(not ng, reason="ngspice not installed")
def test_montecarlo_seed_reproducible_workers1() -> None:
    """Two runs with the same seed produce identical samples and outputs."""

    # Voltage divider
    def build() -> tuple[Circuit, Resistor, Resistor]:
        c = Circuit("mc_repro")
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
        return c, R1, R2

    c1, R1a, R2a = build()
    c2, R1b, R2b = build()

    mc1 = monte_carlo(
        c1,
        {R1a: NormalPct(0.05), R2a: NormalPct(0.05)},
        n=10,
        analyses=[AnalysisSpec("op", {})],
        engine="ngspice",
        seed=777,
        workers=1,
        progress=False,
    )
    mc2 = monte_carlo(
        c2,
        {R1b: NormalPct(0.05), R2b: NormalPct(0.05)},
        n=10,
        analyses=[AnalysisSpec("op", {})],
        engine="ngspice",
        seed=777,
        workers=1,
        progress=False,
    )

    assert mc1.samples == mc2.samples
    v1 = [float(r.traces["V(n1)"].values[-1]) for r in mc1.runs]
    v2 = [float(r.traces["V(n1)"].values[-1]) for r in mc2.runs]
    assert np.allclose(v1, v2, rtol=0, atol=1e-12)


@pytest.mark.skipif(not ng, reason="ngspice not installed")
def test_montecarlo_tran_rc_monotonic_vs_resistance() -> None:
    """
    RC low-pass with input step: Vout(t) = Vin*(1-exp(-t/RC)). For a fixed sample time,
    larger R (→ larger tau) yields smaller Vout. We verify monotonicity for MC samples.
    """
    c = Circuit("mc_tran_rc")
    vin = Net("vin")
    vout = Net("vout")
    # Step source 0 -> 1 V, fast edges compared to RC.
    # Use a pulse source for a clean rising edge without extra directives.
    from spicelab.core.components import Vpulse

    VP = Vpulse("1p", 0.0, 1.0, 0.0, 1e-6, 1e-6, 1.0, 2.0)
    R1 = Resistor("1", 1000.0)
    C1 = Capacitor("1", 1e-6)
    c.add(VP, R1, C1)
    c.connect(VP.ports[0], vin)
    c.connect(VP.ports[1], GND)
    c.connect(R1.ports[0], vin)
    c.connect(R1.ports[1], vout)
    c.connect(C1.ports[0], vout)
    c.connect(C1.ports[1], GND)

    mc = monte_carlo(
        c,
        mapping={R1: NormalPct(0.1)},  # 10% sigma on R
        n=12,
        analyses=[AnalysisSpec("tran", {"tstep": "0.1ms", "tstop": "5ms"})],
        engine="ngspice",
        seed=42,
        workers=1,
        progress=False,
    )

    # Sample v(vout) at t=1 ms via the helper
    df = mc.to_dataframe(metric=None, y=["V(vout)"], sample_at=1e-3)
    v_at = df["V(vout)"].to_numpy(dtype=float)
    r_vals = np.array([s["Resistor.1"] for s in mc.samples], dtype=float)
    order = np.argsort(r_vals)
    v_sorted = v_at[order]
    # Expect non-increasing (allow tiny numerical tolerance)
    diffs = np.diff(v_sorted)
    assert np.all(diffs <= 1e-6)
    # And at least one strictly decreasing pair (allow tiny tolerance)
    assert np.any(diffs < -1e-12)
