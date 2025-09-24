from __future__ import annotations

import importlib
import shutil

import pytest
from spicelab.analysis.sweep_grid import run_value_sweep
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.core.types import AnalysisSpec


@pytest.mark.skipif(not shutil.which("ngspice"), reason="ngspice not installed")
def test_run_value_sweep_smoke() -> None:
    try:
        importlib.import_module("xarray")
    except Exception:
        pytest.skip("xarray not installed")

    c = Circuit("rc_sweep_test")
    vin, vout = Net("vin"), Net("vout")
    V1 = Vdc("1", 5.0)
    R1 = Resistor("1", "1k")
    C1 = Capacitor("1", "100n")
    c.add(V1, R1, C1)
    c.connect(V1.ports[0], vin)
    c.connect(V1.ports[1], GND)
    c.connect(R1.ports[0], vin)
    c.connect(R1.ports[1], vout)
    c.connect(C1.ports[0], vout)
    c.connect(C1.ports[1], GND)

    values = ["1k", "2k"]
    analyses = [AnalysisSpec("tran", {"tstep": 10e-6, "tstop": 1e-3})]
    sweep = run_value_sweep(c, R1, values, analyses, engine="ngspice")
    assert len(sweep.runs) == len(values)
    for run in sweep.runs:
        ds = run.handle.dataset()
        assert ds is not None
        keys = {k.lower() for k in (set(ds.coords.keys()) | set(ds.data_vars.keys()))}
        assert "time" in keys
