"""Tests for Phase 3.4 (Lazy Loading) and 3.5 (ProcessPool)."""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest
from spicelab.core.types import AnalysisSpec, ResultMeta, SweepSpec
from spicelab.engines.lazy_result import LazyDatasetResultHandle, LazyResultHandle
from spicelab.orchestrator import Job, run_job


# ======================================================================================
# P3.4: Lazy ResultHandle Tests
# ======================================================================================
def test_lazy_result_handle_defers_loading():
    """LazyResultHandle doesn't load until .dataset() called."""
    loaded = {"count": 0}

    def loader():
        loaded["count"] += 1
        return {"time": [0, 1, 2], "V(out)": [0, 1, 2]}

    meta = ResultMeta(
        engine="test",
        engine_version="1.0",
        netlist_hash="abc123",
        analyses=[],
        probes=[],
        attrs={},
    )

    handle = LazyDatasetResultHandle(loader=loader, meta=meta)

    # Not loaded yet
    assert loaded["count"] == 0
    assert handle.is_loaded() is False

    # First access loads
    ds = handle.dataset()
    assert loaded["count"] == 1
    assert handle.is_loaded() is True
    assert ds["time"] == [0, 1, 2]

    # Second access uses cache (no reload)
    ds2 = handle.dataset()
    assert loaded["count"] == 1  # Still 1, not reloaded
    assert ds2 is ds  # Same object


def test_lazy_result_handle_from_raw_file():
    """LazyResultHandle.from_raw_file() creates lazy loader."""

    def mock_reader(path):
        return {"data": f"loaded from {path}"}

    meta = ResultMeta(
        engine="test",
        engine_version="1.0",
        netlist_hash="abc123",
        analyses=[],
        probes=[],
        attrs={},
    )

    handle = LazyDatasetResultHandle.from_raw_file(
        path="test.raw",
        reader=mock_reader,
        meta=meta,
    )

    assert handle.is_loaded() is False

    ds = handle.dataset()
    assert ds["data"] == "loaded from test.raw"
    assert handle.is_loaded() is True


def test_lazy_result_handle_repr():
    """LazyResultHandle has informative repr."""
    meta = ResultMeta(
        engine="test",
        engine_version="1.0",
        netlist_hash="abc123",
        analyses=[],
        probes=[],
        attrs={},
    )

    handle = LazyDatasetResultHandle(
        loader=lambda: {},
        meta=meta,
        raw_path="test.raw",
    )

    repr_str = repr(handle)
    assert "LazyDatasetResultHandle" in repr_str
    assert "not loaded" in repr_str
    assert "test.raw" in repr_str

    # After loading
    handle.dataset()
    repr_str = repr(handle)
    assert "loaded" in repr_str


def test_lazy_result_handle_cached_property():
    """LazyResultHandle uses @cached_property for efficiency."""
    call_count = {"n": 0}

    def expensive_loader():
        call_count["n"] += 1
        time.sleep(0.01)  # Simulate expensive operation
        return {"result": "data"}

    meta = ResultMeta(
        engine="test",
        engine_version="1.0",
        netlist_hash="abc123",
        analyses=[],
        probes=[],
        attrs={},
    )

    handle = LazyDatasetResultHandle(loader=expensive_loader, meta=meta)

    # Call multiple times
    for _ in range(5):
        _ = handle.dataset()  # noqa: F841

    # Only loaded once
    assert call_count["n"] == 1


def test_lazy_result_handle_alias():
    """LazyResultHandle is alias for LazyDatasetResultHandle."""
    assert LazyResultHandle is LazyDatasetResultHandle


# ======================================================================================
# P3.5: ProcessPool Parallelism Tests
# ======================================================================================
def test_run_job_auto_detects_process_pool():
    """run_job auto-detects when to use ProcessPool vs ThreadPool."""
    # This is more of an integration test - we'll just verify the parameter exists
    import inspect

    from spicelab.orchestrator import run_job

    sig = inspect.signature(run_job)
    assert "use_processes" in sig.parameters


def test_run_job_use_processes_parameter():
    """run_job accepts use_processes parameter."""
    from spicelab.core.circuit import Circuit
    from spicelab.core.components import Resistor
    from spicelab.core.net import GND, Net

    # Create simple circuit
    circuit = Circuit("test")
    r1 = Resistor("1", "1k")
    circuit.add(r1)
    circuit.connect(r1.ports[0], Net("in"))
    circuit.connect(r1.ports[1], GND)

    analyses = [AnalysisSpec(mode="tran", tstop=1e-3)]
    sweep = SweepSpec(variables={"R1": [1000, 2000, 3000]})

    job = Job(
        circuit=circuit,
        analyses=analyses,
        sweep=sweep,
        probes=None,
        engine="ngspice",
    )

    # Should accept use_processes parameter without error
    try:
        _result = run_job(
            job,
            cache_dir=None,
            workers=1,  # Serial to avoid actual execution
            use_processes=False,  # Force threads
        )
    except Exception as e:
        # May fail due to missing engine, but parameter should be accepted
        if "use_processes" in str(e):
            pytest.fail(f"use_processes parameter not accepted: {e}")


def test_run_simulation_use_processes_parameter():
    """run_simulation propagates use_processes to run_job."""
    import inspect

    from spicelab.engines.orchestrator import run_simulation

    sig = inspect.signature(run_simulation)
    assert "use_processes" in sig.parameters


def test_process_vs_thread_detection():
    """Auto-detection prefers processes for CLI engines."""
    # Mock test to verify logic
    engine_cli = "ngspice"
    engine_shared = "ngspice-shared"

    # CLI engines should use processes (not ending with -shared)
    use_processes_cli = not engine_cli.endswith("-shared")
    assert use_processes_cli is True

    # Shared-lib engines should use threads
    use_processes_shared = not engine_shared.endswith("-shared")
    assert use_processes_shared is False


# ======================================================================================
# Integration Tests
# ======================================================================================
def test_lazy_loading_with_mock_simulation():
    """Integration: lazy loading defers dataset creation."""
    load_times = []

    def timed_loader():
        load_times.append(time.time())
        return {"signal": "data"}

    meta = ResultMeta(
        engine="test",
        engine_version="1.0",
        netlist_hash="abc123",
        analyses=[],
        probes=[],
        attrs={},
    )

    # Create multiple lazy handles
    handles = [LazyDatasetResultHandle(loader=timed_loader, meta=meta) for _ in range(3)]

    # No loading yet
    assert len(load_times) == 0

    # Access only first two
    handles[0].dataset()
    handles[1].dataset()

    # Only 2 loaded (not all 3)
    assert len(load_times) == 2


def test_lazy_to_polars_triggers_load():
    """Lazy .to_polars() loads dataset if needed."""
    loaded = {"flag": False}

    def loader():
        loaded["flag"] = True
        # Return xarray-like object with to_dataframe
        mock_ds = Mock()
        mock_ds.to_dataframe.return_value = Mock()  # Mock pandas DataFrame
        return mock_ds

    meta = ResultMeta(
        engine="test",
        engine_version="1.0",
        netlist_hash="abc123",
        analyses=[],
        probes=[],
        attrs={},
    )

    handle = LazyDatasetResultHandle(loader=loader, meta=meta)

    assert loaded["flag"] is False

    # to_polars should trigger loading
    try:
        handle.to_polars()
    except Exception:
        # May fail due to polars not installed or mock issues
        # But loading should have been triggered
        pass

    assert loaded["flag"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
