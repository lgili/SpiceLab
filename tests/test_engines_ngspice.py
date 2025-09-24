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
    keys = {k.lower() for k in ds.data_vars.keys()}
    assert "freq" in keys or "frequency" in keys
    assert any(k.startswith("v(") for k in keys)
    assert handle.attrs().get("engine") == "ngspice"
