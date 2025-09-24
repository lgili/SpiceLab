from __future__ import annotations

import importlib
import os
import shutil

import pytest
from spicelab.core.circuit import Circuit
from spicelab.core.components import VA, Resistor
from spicelab.core.net import GND
from spicelab.core.types import AnalysisSpec
from spicelab.engines.ltspice import LtSpiceSimulator
from spicelab.engines.xyce import XyceSimulator


def _has_ltspice() -> bool:
    if os.environ.get("SPICELAB_LTSPICE"):
        return True
    return bool(shutil.which("ltspice") or shutil.which("LTspice") or shutil.which("XVIIx64.exe"))


def _has_xyce() -> bool:
    if os.environ.get("SPICELAB_XYCE"):
        return True
    return bool(shutil.which("Xyce") or shutil.which("xyce"))


@pytest.mark.skipif(not _has_ltspice(), reason="LTspice not installed")
def test_ltspice_simulator_tran_smoke() -> None:
    try:
        xr = importlib.import_module("xarray")
    except Exception:
        pytest.skip("xarray not installed")

    c = Circuit("engine_tran_lt")
    V1 = VA(label="1.0")
    R1 = Resistor("1", "1k")
    c.add(V1, R1)
    c.connect(V1.ports[0], R1.ports[0])
    c.connect(V1.ports[1], GND)
    c.connect(R1.ports[1], GND)

    spec = AnalysisSpec("tran", {"tstep": 1e-6, "tstop": 1e-3})
    sim = LtSpiceSimulator()
    handle = sim.run(c, [spec])
    ds = handle.dataset()
    assert isinstance(ds, xr.Dataset)
    keys = {k.lower() for k in (set(ds.data_vars.keys()) | set(ds.coords.keys()))}
    assert "time" in keys
    assert any(k.startswith("v(") for k in keys)
    assert handle.attrs().get("engine") == "ltspice"


@pytest.mark.skipif(not _has_xyce(), reason="Xyce not installed")
def test_xyce_simulator_ac_smoke() -> None:
    try:
        xr = importlib.import_module("xarray")
    except Exception:
        pytest.skip("xarray not installed")

    c = Circuit("engine_ac_xy")
    V1 = VA(ac_mag=1.0)
    R1 = Resistor("1", "1k")
    c.add(V1, R1)
    c.connect(V1.ports[0], R1.ports[0])
    c.connect(V1.ports[1], GND)
    c.connect(R1.ports[1], GND)

    spec = AnalysisSpec("ac", {"sweep_type": "dec", "n": 5, "fstart": 10.0, "fstop": 1e3})
    sim = XyceSimulator()
    handle = sim.run(c, [spec])
    ds = handle.dataset()
    assert isinstance(ds, xr.Dataset)
    keys = {k.lower() for k in (set(ds.data_vars.keys()) | set(ds.coords.keys()))}
    assert "freq" in keys or "frequency" in keys
    assert any(k.startswith("v(") for k in keys)
    assert handle.attrs().get("engine") == "xyce"
