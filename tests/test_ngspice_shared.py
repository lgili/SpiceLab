from __future__ import annotations

import ctypes
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND
from spicelab.core.types import AnalysisSpec
from spicelab.engines.exceptions import EngineSharedLibraryNotFound
from spicelab.engines.ngspice_shared import NgSpiceSharedSimulator
from spicelab.spice.ngspice_shared import (
    ExternalSourceRequest,
    NgSpiceSharedAdapter,
    SharedAdapterResult,
    SharedCallbacks,
    SharedTranPoint,
)

xr = cast(Any, pytest.importorskip("xarray"))


@dataclass
class _FakeSharedBackend:
    steps: int = 5

    def run(
        self,
        netlist: str,
        directives: Sequence[str],
        callbacks: SharedCallbacks | None,
        external_sources: Mapping[str, Callable[[ExternalSourceRequest], float]] | None,
    ) -> SharedAdapterResult:
        times = np.linspace(0.0, 1e-6, self.steps)
        value = 0.0
        values: list[float] = []
        log: list[str] = []
        if callbacks and callbacks.on_init:
            callbacks.on_init()
        for t in times:
            request_values = {"V(out)": value}
            ctrl = 0.0
            if external_sources:
                for name, fn in external_sources.items():
                    ctrl = fn(ExternalSourceRequest(name=name, time=t, values=request_values))
                    request_values[name] = ctrl
            value += ctrl * 0.5
            values.append(value)
            if callbacks and callbacks.on_tran_point:
                callbacks.on_tran_point(SharedTranPoint(time=float(t), values={"V(out)": value}))
        if callbacks and callbacks.on_exit:
            callbacks.on_exit(0)
        ds = xr.Dataset({"V(out)": ("time", np.array(values))}, coords={"time": times})
        return SharedAdapterResult(dataset=ds, log_lines=log, stdout="", stderr="")


def _simple_circuit() -> Circuit:
    circuit = Circuit("shared_rc")
    v1 = Vdc("VIN", 1.0)
    r1 = Resistor("R1", "1k")
    circuit.add(v1, r1)
    circuit.connect(v1.ports[0], r1.ports[0])
    circuit.connect(r1.ports[1], GND)
    circuit.connect(v1.ports[1], GND)
    return circuit


def test_ngspice_shared_simulator_invokes_callbacks() -> None:
    backend = _FakeSharedBackend()
    adapter = NgSpiceSharedAdapter(backend)
    sim = NgSpiceSharedSimulator(adapter)
    circuit = _simple_circuit()

    seen_times: list[float] = []

    def on_tran(point: SharedTranPoint) -> None:
        seen_times.append(point.time)

    controller_calls: list[float] = []

    def controller(req: ExternalSourceRequest) -> float:
        controller_calls.append(req.time)
        return 1.0 - req.values.get("V(out)", 0.0)

    analysis = AnalysisSpec(
        mode="tran",
        args={
            "tstop": 1e-6,
            "tstep": 2e-7,
            "callbacks": {"on_tran_point": on_tran},
            "external_sources": {"Vctrl": controller},
        },
    )

    handle = sim.run(circuit, [analysis])
    ds = handle.dataset()
    assert "V(out)" in ds
    assert "time" in ds.coords
    assert seen_times  # callbacks fired
    assert controller_calls  # external source invoked
    attrs = handle.attrs()
    assert attrs["engine"] == "ngspice-shared"


def test_ngspice_shared_adapter_without_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "spicelab.spice.ngspice_shared_backend._candidate_paths",
        lambda: ["/missing/libngspice.so"],
    )
    adapter = NgSpiceSharedAdapter()
    with pytest.raises(EngineSharedLibraryNotFound):
        adapter.run("* empty", [".tran 1n 1u"])


def test_ctypes_backend_external_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    from spicelab.spice.ngspice_shared_backend import (
        CtypesNgSpiceSharedBackend,
        VecValues,
        VecValuesAll,
    )

    backend = object.__new__(CtypesNgSpiceSharedBackend)
    backend._external_sources = {"VCTRL": lambda req: req.time + req.values.get("V(out)", 0.0)}
    backend._last_source_values = {}
    backend._logs = []
    commands: list[str] = []

    def fake_command(cmd: str) -> None:
        commands.append(cmd)

    cast(Any, backend)._command = fake_command

    vec_time = VecValues()
    vec_time.name = b"time"
    vec_time.creal = ctypes.c_double(2e-6).value

    vec_vout = VecValues()
    vec_vout.name = b"V(out)"
    vec_vout.creal = ctypes.c_double(0.2).value

    array_type = ctypes.POINTER(VecValues) * 2
    vec_array = array_type(ctypes.pointer(vec_time), ctypes.pointer(vec_vout))
    vec_all = VecValuesAll(veccount=2, vecindex=2, vecsa=vec_array)

    backend._send_data(ctypes.pointer(vec_all), 0, 0, None)

    assert commands  # alter command issued
    assert any(cmd.startswith("alter @VCTRL[dc]") for cmd in commands)
