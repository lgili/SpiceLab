from __future__ import annotations

import importlib
import shutil

import pytest
from spicelab.core.circuit import Circuit
from spicelab.core.components import VA, Resistor
from spicelab.core.net import GND
from spicelab.core.types import AnalysisSpec
from spicelab.engines.ngspice import NgSpiceSimulator


@pytest.mark.skipif(not shutil.which("ngspice"), reason="ngspice not installed")
def test_ngspice_simulator_ac_smoke() -> None:
    try:
        xr = importlib.import_module("xarray")
    except Exception:
        pytest.skip("xarray not installed")

    c = Circuit("engine_ac")
    V1 = VA(ac_mag=1.0)
    R1 = Resistor("1", "1k")
    c.add(V1, R1)
    c.connect(V1.ports[0], R1.ports[0])  # node vin
    c.connect(V1.ports[1], GND)
    c.connect(R1.ports[1], GND)

    spec = AnalysisSpec("ac", {"sweep_type": "dec", "n": 10, "fstart": 10.0, "fstop": 1e6})
    sim = NgSpiceSimulator()
    handle = sim.run(c, [spec])
    ds = handle.dataset()
    assert isinstance(ds, xr.Dataset)
    keys = {k.lower() for k in (set(ds.data_vars.keys()) | set(ds.coords.keys()))}
    assert "freq" in keys or "frequency" in keys
    assert any(k.startswith("v(") for k in keys)
    assert handle.attrs().get("engine") == "ngspice"


@pytest.mark.skipif(not shutil.which("ngspice"), reason="ngspice not installed")
def test_ngspice_simulator_tran_smoke() -> None:
    try:
        xr = importlib.import_module("xarray")
    except Exception:
        pytest.skip("xarray not installed")

    # RC discharge: V -> R -> C -> GND
    c = Circuit("engine_tran")
    V1 = VA(label="1.0")
    R1 = Resistor("1", "1k")
    c.add(V1, R1)
    c.connect(V1.ports[0], R1.ports[0])  # node vin
    c.connect(V1.ports[1], GND)
    c.connect(R1.ports[1], GND)

    spec = AnalysisSpec("tran", {"tstep": 1e-6, "tstop": 1e-3})
    sim = NgSpiceSimulator()
    handle = sim.run(c, [spec])
    ds = handle.dataset()
    assert isinstance(ds, xr.Dataset)
    keys = {k.lower() for k in (set(ds.data_vars.keys()) | set(ds.coords.keys()))}
    assert "time" in keys
    assert any(k.startswith("v(") for k in keys)
    assert handle.attrs().get("engine") == "ngspice"


@pytest.mark.skipif(not shutil.which("ngspice"), reason="ngspice not installed")
def test_ngspice_simulator_dc_smoke() -> None:
    try:
        xr = importlib.import_module("xarray")
    except Exception:
        pytest.skip("xarray not installed")

    c = Circuit("engine_dc")
    V1 = VA(label="0.0")
    R1 = Resistor("1", "1k")
    c.add(V1, R1)
    c.connect(V1.ports[0], R1.ports[0])  # node vin
    c.connect(V1.ports[1], GND)
    c.connect(R1.ports[1], GND)

    # Sweep the source V1 from 0 to 1 in steps of 0.2
    spec = AnalysisSpec("dc", {"src": V1.ref, "start": 0.0, "stop": 1.0, "step": 0.2})
    sim = NgSpiceSimulator()
    handle = sim.run(c, [spec])
    ds = handle.dataset()
    assert isinstance(ds, xr.Dataset)
    keys = {k.lower() for k in (set(ds.data_vars.keys()) | set(ds.coords.keys()))}
    # NGSpice DC uses the swept source name as the x variable; ensure it's present somehow
    assert any("v(" in k or k == "time" or k == "freq" or "v" in k for k in keys)
    assert any(k.startswith("v(") for k in {k.lower() for k in ds.data_vars.keys()})
    assert handle.attrs().get("engine") == "ngspice"
