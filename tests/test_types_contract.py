from __future__ import annotations

from spicelab.core.types import AnalysisSpec, SweepSpec, circuit_hash, stable_hash


def test_analysis_spec_hash_deterministic() -> None:
    a = AnalysisSpec(mode="tran", args={"step": "1e-6", "stop": "1e-3"})
    b = AnalysisSpec(mode="tran", args={"stop": "1e-3", "step": "1e-6"})
    # Same content, different dict order → same hash
    ha = stable_hash(a)
    hb = stable_hash(b)
    assert ha == hb


def test_circuit_hash_uses_build_netlist() -> None:
    class Dummy:
        def __init__(self, s: str) -> None:
            self.s = s

        def build_netlist(self) -> str:
            return self.s

    h1 = circuit_hash(Dummy("R1 1 0 1k"))
    h2 = circuit_hash(Dummy("R1 1 0 1k"))
    assert h1 == h2


def test_sweep_spec_roundtrip() -> None:
    s = SweepSpec(variables={"R1": [100.0, 200.0, 300.0]})
    assert list(s.variables.keys()) == ["R1"]
