from __future__ import annotations

import pytest
from spicelab.core.types import AnalysisSpec, Probe
from spicelab.engines.orchestrator import JobResult, run_simulation


class DummyCircuit:
    def build_netlist(self) -> str:
        return "R1 1 0 1k\nV1 1 0 1\n.op\n.end"


def test_probes_present_in_result_attrs(monkeypatch: object) -> None:
    # Rely on ngspice presence; if adapter missing we skip gracefully.
    try:
        from spicelab.spice.registry import get_active_adapter  # noqa: F401
    except Exception:  # pragma: no cover
        pytest.skip("ngspice adapter not available")

    circuit = DummyCircuit()
    analyses = [AnalysisSpec("op")]
    probes = [Probe(kind="voltage", target="1")]
    res = run_simulation(circuit, analyses, probes=probes, engine="ngspice")
    if isinstance(res, JobResult):  # pragma: no cover - not expected for OP
        pytest.skip("unexpected sweep result returned")
    attrs = res.attrs()
    assert "probes" in attrs
    assert any(p["kind"] == "voltage" for p in attrs["probes"])  # sanity
