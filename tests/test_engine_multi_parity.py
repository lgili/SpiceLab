from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from spicelab.analysis.sweep_grid import GridResult, run_param_grid
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND
from spicelab.core.types import (
    AnalysisSpec,
    Probe,
    ResultHandle,
    ResultMeta,
    SweepSpec,
    circuit_hash,
)
from spicelab.engines.base import EngineFeatures, Simulator
from spicelab.engines.factory import create_simulator as _create_simulator
from spicelab.engines.result import DatasetResultHandle
from spicelab.io.readers import read_ltspice_raw, read_ngspice_raw, read_xyce_prn


def _load_tran() -> tuple[Any, Any, Any]:
    return (
        read_ngspice_raw("tests/fixtures/rc_tran_ng.raw"),
        read_ltspice_raw("tests/fixtures/rc_tran_lt.raw"),
        read_xyce_prn("tests/fixtures/rc_tran_xy.prn"),
    )


def _load_ac() -> tuple[Any, Any, Any]:
    return (
        read_ngspice_raw("tests/fixtures/rc_ac_ng.raw"),
        read_ltspice_raw("tests/fixtures/rc_ac_lt.raw"),
        read_xyce_prn("tests/fixtures/rc_ac_xy.prn"),
    )


def test_rc_tran_shapes_match_across_engines() -> None:
    ng, lt, xy = _load_tran()
    for ds in (ng, lt, xy):
        assert set(ds.data_vars) >= {"V(out)", "I(R1)"}
        assert "time" in ds
    ref_time = ng["time"].values
    ref_vout = ng["V(out)"].values
    for other in (lt, xy):
        assert other["V(out)"].shape == ref_vout.shape
        assert np.allclose(other["time"], ref_time, atol=1e-12)
        assert np.allclose(other["V(out)"], ref_vout, rtol=1e-2, atol=1e-3)


def test_rc_ac_shapes_match_across_engines() -> None:
    ng, lt, xy = _load_ac()
    for ds in (ng, lt, xy):
        assert "V(out)" in ds.data_vars
        assert "freq" in ds
    ref_freq = ng["freq"].values
    ref_vout = ng["V(out)"].values
    for other in (lt, xy):
        assert other["V(out)"].shape == ref_vout.shape
        assert np.allclose(other["freq"], ref_freq, atol=1e-12)
        assert np.allclose(other["V(out)"], ref_vout, rtol=1e-2, atol=1e-3)


def test_param_grid_engine_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    fixtures = {
        "ngspice": read_ngspice_raw("tests/fixtures/rc_tran_ng.raw"),
        "ltspice": read_ltspice_raw("tests/fixtures/rc_tran_lt.raw"),
        "xyce": read_xyce_prn("tests/fixtures/rc_tran_xy.prn"),
    }

    class _FixtureSimulator(Simulator):
        def __init__(self, engine: str) -> None:
            self._engine = engine
            ds = fixtures[engine]
            self._template = ds
            self._features = EngineFeatures(
                name=f"{engine}-fixture",
                supports_noise=False,
                supports_callbacks=False,
                supports_shared_lib=False,
                supports_verilog_a=False,
                supports_parallel=False,
            )

        def features(self) -> EngineFeatures:
            return self._features

        def run(
            self,
            circuit: object,
            analyses: Sequence[AnalysisSpec],
            sweep: SweepSpec | None = None,
            probes: list[Probe] | None = None,
        ) -> ResultHandle:
            if sweep and sweep.variables:
                raise AssertionError("Fixture simulator expects sweep to be expanded externally")
            ds = self._template.copy(deep=True)
            net_hash = circuit_hash(circuit, extra={"analyses": [spec.mode for spec in analyses]})
            meta = ResultMeta(
                engine=self._engine,
                engine_version="fixture",
                netlist_hash=net_hash,
                analyses=list(analyses),
                probes=list(probes or []),
                attrs={"fixture": True},
            )
            return DatasetResultHandle(ds, meta)

    def _stub_create(engine: str) -> Simulator:
        if engine in fixtures:
            return _FixtureSimulator(engine)
        return _create_simulator(engine)

    monkeypatch.setattr("spicelab.engines.factory.create_simulator", _stub_create)
    monkeypatch.setattr("spicelab.engines.orchestrator.create_simulator", _stub_create)
    monkeypatch.setattr("spicelab.orchestrator.create_simulator", _stub_create)

    combos_expected = None
    baseline_shape = None
    baseline_vars = None

    def _grid_result(engine: str) -> GridResult:
        circuit = Circuit(f"grid_{engine}")
        v1 = Vdc("VIN", 1.0)
        r1 = Resistor("R", "1k")
        c1 = Capacitor("C", "100n")
        circuit.add(v1, r1, c1)
        circuit.connect(v1.ports[0], r1.ports[0])
        circuit.connect(r1.ports[1], c1.ports[0])
        circuit.connect(v1.ports[1], GND)
        circuit.connect(c1.ports[1], GND)

        result = run_param_grid(
            circuit=circuit,
            variables=[
                (v1, [1.0, 5.0]),
                (r1, ["1k", "2k"]),
                (c1, ["100n", "220n"]),
            ],
            analyses=[AnalysisSpec("op", {})],
            engine=engine,
            reuse_cache=False,
        )
        return result

    results = {engine_name: _grid_result(engine_name) for engine_name in fixtures}

    for result in results.values():
        combos = [tuple(sorted(run.combo.items())) for run in result.runs]
        if combos_expected is None:
            combos_expected = combos
        else:
            assert combos == combos_expected

        for run in result.runs:
            ds = run.handle.dataset()
            dims = tuple(sorted(ds.dims.items()))
            vars_present = tuple(
                sorted(
                    (
                        name,
                        ds[name].shape,
                    )
                    for name in ds.data_vars
                    if name.lower() not in {"t", "time", "freq", "frequency"}
                )
            )
            if baseline_shape is None:
                baseline_shape = dims
                baseline_vars = vars_present
            else:
                assert dims == baseline_shape
                assert vars_present == baseline_vars
