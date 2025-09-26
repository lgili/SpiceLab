from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from spicelab.core.circuit import Circuit
from spicelab.core.components import VA, Resistor
from spicelab.core.net import GND
from spicelab.core.types import AnalysisSpec
from spicelab.engines.factory import create_simulator


def _candidate_works(exe: str) -> bool:
    try:
        exe_path = Path(exe)
        if not exe_path.exists():
            return False
        with tempfile.TemporaryDirectory(prefix="xyce_check_") as tmp:
            deck = Path(tmp) / "check.cir"
            deck.write_text(
                "\n".join(
                    [
                        "* spicelab xyce availability probe",
                        "V1 1 0 AC 1",
                        "R1 1 0 1k",
                        ".ac dec 1 1 1",
                        ".print ac format=csv V(1)",
                        ".end",
                    ]
                ),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [exe, str(deck)],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            return proc.returncode == 0 and bool(list(Path(tmp).glob("check.cir*.csv")))
    except Exception:
        return False


def _has_ltspice() -> bool:
    if os.environ.get("SPICELAB_LTSPICE"):
        return True
    return bool(shutil.which("ltspice") or shutil.which("LTspice") or shutil.which("XVIIx64.exe"))


def _has_xyce() -> bool:
    env_exe = os.environ.get("SPICELAB_XYCE") or os.environ.get("XYCE_EXE")
    if env_exe and _candidate_works(env_exe):
        return True

    for candidate in (shutil.which("Xyce"), shutil.which("xyce")):
        if candidate and _candidate_works(candidate):
            return True
    return False


@pytest.mark.skipif(not _has_ltspice(), reason="LTspice not installed")
@pytest.mark.engine
@pytest.mark.parametrize("engine_name", ["ltspice", "ltspice-cli"])
def test_ltspice_simulator_tran_smoke(engine_name: str) -> None:
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
    sim = create_simulator(engine_name)
    handle = sim.run(c, [spec])
    ds = handle.dataset()
    assert isinstance(ds, xr.Dataset)
    keys = {k.lower() for k in (set(ds.data_vars.keys()) | set(ds.coords.keys()))}
    assert "time" in keys
    assert any(k.startswith("v(") for k in keys)
    assert handle.attrs().get("engine") == "ltspice"
    feats = sim.features()
    assert feats.name == "ltspice-cli"
    assert not feats.supports_parallel
    assert feats.supports_noise is True


@pytest.mark.skipif(not _has_xyce(), reason="Xyce not installed")
@pytest.mark.engine
@pytest.mark.parametrize("engine_name", ["xyce", "xyce-cli"])
def test_xyce_simulator_ac_smoke(engine_name: str) -> None:
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
    sim = create_simulator(engine_name)
    handle = sim.run(c, [spec])
    ds = handle.dataset()
    assert isinstance(ds, xr.Dataset)
    keys = {k.lower() for k in (set(ds.data_vars.keys()) | set(ds.coords.keys()))}
    assert "freq" in keys or "frequency" in keys
    assert any(k.startswith("v(") for k in keys)
    assert handle.attrs().get("engine") == "xyce"
    feats = sim.features()
    assert feats.name == "xyce-cli"
    assert feats.supports_parallel
    assert feats.supports_noise is True
    assert feats.supports_verilog_a is True
