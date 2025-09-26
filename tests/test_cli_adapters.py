from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from spicelab.core.types import AnalysisSpec
from spicelab.engines.exceptions import EngineBinaryNotFound
from spicelab.engines.ltspice_cli import LtSpiceCLISimulator
from spicelab.engines.xyce_cli import XyceCLISimulator
from spicelab.spice.base import RunArtifacts, RunResult
from spicelab.spice.ltspice_cli import (
    DEFAULT_ADAPTER as LT_DEFAULT_ADAPTER,
)
from spicelab.spice.ltspice_cli import (
    _strip_final_end,
    _which_ltspice,
)
from spicelab.spice.ltspice_cli import (
    run_directives as lt_run_directives,
)
from spicelab.spice.xyce_cli import DEFAULT_ADAPTER as XY_DEFAULT_ADAPTER
from spicelab.spice.xyce_cli import _which_xyce
from spicelab.spice.xyce_cli import run_directives as xy_run_directives

xr = cast(Any, pytest.importorskip("xarray"))


class DummyCircuit:
    def __init__(self, netlist: str = "* demo") -> None:
        self._net = netlist

    def build_netlist(self) -> str:
        return self._net


@pytest.fixture(autouse=True)
def restore_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPICELAB_LTSPICE", raising=False)
    monkeypatch.delenv("LTSPICE_EXE", raising=False)
    monkeypatch.delenv("SPICELAB_XYCE", raising=False)
    monkeypatch.delenv("XYCE_EXE", raising=False)


def test_strip_final_end_removes_extra_end_cards() -> None:
    text = "R1 1 0 1k\n.end\n\n"
    assert _strip_final_end(text) == "R1 1 0 1k"


def test_ltspice_run_directives_creates_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = tmp_path / "ltspice"
    exe.write_text("", encoding="utf-8")
    monkeypatch.setenv("SPICELAB_LTSPICE", str(exe))

    def fake_run(cmd: Iterable[str], cwd: str, capture_output: bool, text: bool) -> SimpleNamespace:
        cmd_list = list(cmd)
        assert Path(cmd_list[0]) == exe
        raw = Path(cwd) / "deck.raw"
        raw.write_text("!raw", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("spicelab.spice.ltspice_cli.subprocess.run", fake_run)
    result = lt_run_directives("V 1 0 5", [".op"], title="unit", keep_workdir=False)
    assert result.returncode == 0
    assert result.artifacts.raw_path and Path(result.artifacts.raw_path).exists()
    log_content = Path(result.artifacts.log_path).read_text(encoding="utf-8")
    assert "ok" in log_content


def test_ltspice_missing_binary_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(os.environ, "SPICELAB_LTSPICE", str(Path("/missing/bin/lt")))
    monkeypatch.setattr("spicelab.spice.ltspice_cli.Path.exists", lambda self: False)
    monkeypatch.setattr("spicelab.spice.ltspice_cli.shutil.which", lambda name: None)
    with pytest.raises(EngineBinaryNotFound):
        _which_ltspice()


def test_ltspice_simulator_success(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = xr.Dataset(
        {"V(out)": ("vsweep", np.array([1.0, 2.0]))},
        coords={"vsweep": np.array([0.0, 1.0])},
    )

    def fake_read(path: str) -> Any:  # pragma: no cover - deterministic stub
        assert path == "raw"
        return ds

    monkeypatch.setattr("spicelab.engines.ltspice_cli.read_ltspice_raw", fake_read)

    def fake_run_directives(netlist: str, directives: list[str]) -> RunResult:
        assert directives[0].startswith(".dc")
        return RunResult(
            artifacts=RunArtifacts(
                netlist_path="deck.cir",
                log_path="lt.log",
                raw_path="raw",
                workdir="work",
            ),
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(LT_DEFAULT_ADAPTER, "run_directives", fake_run_directives)
    circuit = DummyCircuit("R1 1 0 1k")
    sim = LtSpiceCLISimulator()
    handle = sim.run(
        circuit,
        [AnalysisSpec(mode="dc", args={"src": "VIN", "start": 0.0, "stop": 1.0, "step": 0.5})],
    )
    data = handle.dataset()
    assert "sweep" in data.coords
    attrs = handle.attrs()
    assert attrs["engine"] == "ltspice"
    assert attrs["analysis_modes"] == ["dc"]
    assert attrs["dc_sweep_label"] == "vsweep"


def test_xyce_run_directives_prefers_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = tmp_path / "xyce"
    exe.write_text("", encoding="utf-8")
    monkeypatch.setenv("SPICELAB_XYCE", str(exe))

    def fake_run(cmd: Iterable[str], cwd: str, capture_output: bool, text: bool) -> SimpleNamespace:
        workdir = Path(cwd)
        deck_text = (workdir / "deck.cir").read_text(encoding="utf-8")
        assert deck_text.lower().count(".end") == 1
        assert deck_text.splitlines()[-1].strip().lower() == ".end"
        assert ".op" in deck_text
        (workdir / "deck.cir.csv").write_text("csv", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="stdout", stderr="err")

    monkeypatch.setattr("spicelab.spice.xyce_cli.subprocess.run", fake_run)
    result = xy_run_directives("R 1 0 1k\n.end\n", [".op"], keep_workdir=False)
    assert result.artifacts.raw_path and result.artifacts.raw_path.endswith(".csv")


def test_xyce_run_directives_detects_fd_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = tmp_path / "xyce"
    exe.write_text("", encoding="utf-8")
    monkeypatch.setenv("SPICELAB_XYCE", str(exe))

    def fake_run(cmd: Iterable[str], cwd: str, capture_output: bool, text: bool) -> SimpleNamespace:
        workdir = Path(cwd)
        (workdir / "deck.cir.FD.csv").write_text("freq,data", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="stdout", stderr="err")

    monkeypatch.setattr("spicelab.spice.xyce_cli.subprocess.run", fake_run)
    result = xy_run_directives("R 1 0 1k\n.end\n", [".ac dec 5 1 10"], keep_workdir=False)
    assert result.artifacts.raw_path and result.artifacts.raw_path.endswith(".FD.csv")


def test_xyce_simulator_loads_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = xr.Dataset(
        {"V(out)": ("time", np.array([0.0, 1.0]))},
        coords={"time": np.array([0.0, 1.0])},
    )

    def fake_read(path: str) -> Any:
        assert path == "data.tbl"
        return ds

    monkeypatch.setattr("spicelab.engines.xyce_cli.read_xyce_table", fake_read)

    def fake_run_directives(netlist: str, directives: list[str]) -> RunResult:
        assert directives[0] == ".op"
        assert any(d.startswith(".print op format=csv") for d in directives[1:])
        return RunResult(
            artifacts=RunArtifacts(
                netlist_path="deck.cir",
                log_path="xy.log",
                raw_path="data.tbl",
                workdir="work",
            ),
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(XY_DEFAULT_ADAPTER, "run_directives", fake_run_directives)
    sim = XyceCLISimulator()
    handle = sim.run(DummyCircuit("R"), [AnalysisSpec(mode="op", args={})])
    data = handle.dataset()
    assert "time" in data


def test_xyce_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(os.environ, "SPICELAB_XYCE", str(Path("/missing/xyce")))
    monkeypatch.setattr("spicelab.spice.xyce_cli.Path.exists", lambda self: False)
    monkeypatch.setattr("spicelab.spice.xyce_cli.shutil.which", lambda name: None)
    with pytest.raises(EngineBinaryNotFound):
        _which_xyce()
