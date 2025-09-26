from __future__ import annotations

import importlib
import math
import random as _random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from spicelab.analysis.montecarlo import (
    Dist,
    LogNormalPct,
    MonteCarloResult,
    NormalPct,
    TriangularPct,
    UniformAbs,
    UniformPct,
    _as_float,
    _dataset_to_traceset,
    _handle_to_analysis_result,
    monte_carlo,
)
from spicelab.analysis.result import AnalysisResult
from spicelab.core.circuit import Circuit
from spicelab.core.components import Component, Resistor
from spicelab.core.types import AnalysisSpec, ResultMeta
from spicelab.engines.result import DatasetResultHandle
from spicelab.io.raw_reader import Trace, TraceSet
from spicelab.orchestrator import Job, JobResult, JobRun
from spicelab.spice.base import RunArtifacts, RunResult

xr = cast(Any, pytest.importorskip("xarray"))


def _run_result_obj() -> RunResult:
    return RunResult(
        artifacts=RunArtifacts(netlist_path="", log_path="", raw_path=None, workdir=None),
        returncode=0,
        stdout="",
        stderr="",
    )


def _analysis_result(value: float) -> AnalysisResult:
    traces = TraceSet(
        [
            Trace("time", "s", np.array([0.0, 1.0])),
            Trace("V(out)", None, np.array([value, value + 1.0])),
        ]
    )
    return AnalysisResult(run=_run_result_obj(), traces=traces)


def test_monte_carlo_result_to_dataframe_with_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    runs = [
        AnalysisResult(
            run=_run_result_obj(),
            traces=TraceSet(
                [
                    Trace("time", "s", np.array([0.0, 1.0, 2.0])),
                    Trace("V(out)", None, np.array([0.0, 0.5, 1.0])),
                ]
            ),
        ),
        AnalysisResult(
            run=_run_result_obj(),
            traces=TraceSet(
                [
                    Trace("time", "s", np.array([0.0, 1.0, 2.0])),
                    Trace("V(out)", None, np.array([0.0, 1.5, 2.0])),
                ]
            ),
        ),
    ]
    result = MonteCarloResult(
        samples=[{"R": 100.0}, {"R": 200.0}],
        runs=runs,
    )
    metrics = {
        "span": lambda r: float(r.traces["V(out)"].values[-1] - r.traces["V(out)"].values[0])
    }
    df = result.to_dataframe(metric=metrics, param_prefix="p_", y=["V(out)"], sample_at=1.0)
    assert "span" in df.columns
    assert "p_R" in df.columns

    def fake_import(name: str) -> Any:
        if name == "pandas":
            raise ImportError("no pandas")
        return importlib.import_module(name)

    monkeypatch.setattr("spicelab.analysis.montecarlo.importlib.import_module", fake_import)
    fallback = result.to_dataframe(metric=None)
    assert fallback.columns and "R" in fallback.columns


def test_monte_carlo_result_csv_helpers(tmp_path: Path) -> None:
    rs = MonteCarloResult(
        samples=[{"R": 1.0}], runs=[_analysis_result(0.5)], mapping_manifest=[("R", 1.0, "Normal")]
    )
    out = tmp_path / "metrics.csv"
    rs.to_csv(str(out), columns=["R", "trial"])
    assert out.exists()
    samples_csv = tmp_path / "samples.csv"
    rs.save_samples_csv(str(samples_csv))
    assert samples_csv.exists()
    manifest_csv = tmp_path / "manifest.csv"
    rs.save_manifest_csv(str(manifest_csv))
    assert manifest_csv.exists()


def test_dataset_to_traceset_without_coords() -> None:
    ds = xr.Dataset({"V(out)": ("points", np.array([1.0, 2.0, 3.0]))})
    traces = _dataset_to_traceset(ds)
    assert isinstance(traces, TraceSet)
    assert "V(out)" in traces.names


def test_handle_to_analysis_result_builds_analysis() -> None:
    ds = xr.Dataset(
        {"V(out)": ("time", np.array([0.0, 1.0]))}, coords={"time": np.array([0.0, 1.0])}
    )
    meta = ResultMeta(
        engine="ngspice", analyses=[AnalysisSpec(mode="op", args={})], attrs={"raw_path": "raw.raw"}
    )
    handle = DatasetResultHandle(ds, meta)
    analysis = _handle_to_analysis_result(handle)
    assert isinstance(analysis, AnalysisResult)
    assert analysis.traces.names[0] == "time"


def test_as_float_handles_strings() -> None:
    assert _as_float("1k") == pytest.approx(1000.0)


def test_monte_carlo_job_path(monkeypatch: pytest.MonkeyPatch) -> None:
    circuit = Circuit("mc")
    r1 = Resistor("R1", "1k")
    circuit.add(r1)
    mapping: dict[Component, Dist] = {r1: UniformPct(0.0)}

    dataset = xr.Dataset(
        {"V(out)": ("time", np.array([0.0, 1.0]))}, coords={"time": np.array([0.0, 1.0])}
    )
    meta = ResultMeta(engine="ngspice", analyses=[AnalysisSpec(mode="op", args={})])
    handle = DatasetResultHandle(dataset, meta)
    job = Job(circuit=circuit, analyses=[AnalysisSpec(mode="op", args={})])
    job_result = JobResult(
        job=job, runs=[JobRun(combo={"R1": "1k"}, handle=handle)], cache_dir=None
    )

    captured: dict[str, Any] = {}

    def fake_run_job(job: Job, **kwargs: Any) -> JobResult:
        captured["job"] = job
        captured["kwargs"] = kwargs
        return job_result

    monkeypatch.setattr("spicelab.analysis.montecarlo.run_job", fake_run_job)

    def progress_cb(done: int, total: int) -> None:
        pass

    mc = monte_carlo(
        circuit,
        mapping,
        n=1,
        analyses=[AnalysisSpec(mode="op", args={})],
        engine="ngspice",
        cache_dir=None,
        workers=2,
        progress=progress_cb,
    )
    assert mc.result_handles()
    assert captured["kwargs"]["workers"] == 2
    assert captured["kwargs"]["progress"] is progress_cb


@pytest.mark.parametrize(
    "dist_cls, kwargs",
    [
        (NormalPct, {"sigma_pct": 0.1}),
        (LogNormalPct, {"sigma_pct": 0.1}),
        (UniformPct, {"pct": 0.1}),
        (UniformAbs, {"delta": 5.0}),
        (TriangularPct, {"pct": 0.2}),
    ],
)
def test_distributions_sample(dist_cls: type[Dist], kwargs: dict[str, float]) -> None:
    dist = dist_cls(**kwargs)
    value = dist.sample(100.0, _random.Random(123))
    assert math.isfinite(float(value))


def test_distributions_validation() -> None:
    with pytest.raises(ValueError):
        NormalPct(-0.1)
    with pytest.raises(ValueError):
        UniformPct(-0.1)
    with pytest.raises(ValueError):
        UniformAbs(-1.0)
    with pytest.raises(ValueError):
        TriangularPct(-0.5)


def test_dataset_to_traceset_scalar_variable() -> None:
    ds = xr.Dataset(
        {"V(out)": ((), 5.0)},
        coords={"time": ("time", np.array([0.0, 1.0]))},
    )
    traces = _dataset_to_traceset(ds)
    assert traces["V(out)"].values.tolist() == [5.0, 5.0]


def test_dataset_to_traceset_misaligned_raises() -> None:
    class _StubData:
        def __init__(self, arr: np.ndarray) -> None:
            self.values = arr

    class _StubDataset:
        def __init__(self) -> None:
            self.coords = {"time": SimpleNamespace(values=np.array([0.0, 1.0, 2.0]), name="time")}
            self.data_vars = {"bad": _StubData(np.ones((2, 2)))}

        def __getitem__(self, key: str) -> _StubData:
            return self.data_vars[key]

    ds = _StubDataset()
    with pytest.raises(ValueError):
        _dataset_to_traceset(ds)


def test_monte_carlo_requires_analysis() -> None:
    circuit = Circuit("mc")
    r = Resistor("R1", "1k")
    circuit.add(r)
    mapping: dict[Component, Dist] = {r: UniformPct(0.1)}
    with pytest.raises(ValueError, match="analyses"):
        monte_carlo(circuit, mapping, n=1)


def test_monte_carlo_zero_runs_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    circuit = Circuit("mc")
    r = Resistor("R1", "1k")
    circuit.add(r)
    mapping: dict[Component, Dist] = {r: UniformPct(0.0)}

    job = Job(circuit=circuit, analyses=[AnalysisSpec(mode="op", args={})])
    job_result = JobResult(job=job, runs=[], cache_dir=None)
    monkeypatch.setattr("spicelab.analysis.montecarlo.run_job", lambda *args, **kwargs: job_result)

    result = monte_carlo(
        circuit,
        mapping,
        n=0,
        analyses=[AnalysisSpec(mode="op", args={})],
    )
    assert result.runs == []
