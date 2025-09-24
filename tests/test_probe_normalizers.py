from spicelab.core.types import Probe, ensure_probe, ensure_probes


def test_ensure_probe_from_string_voltage() -> None:
    p = ensure_probe("V(out)")
    assert isinstance(p, Probe)
    assert p.kind == "voltage" and p.target == "out"


def test_ensure_probe_from_string_current() -> None:
    p = ensure_probe("i(R1)")
    assert p.kind == "current" and p.target == "R1"


def test_ensure_probe_from_dict() -> None:
    p = ensure_probe({"kind": "voltage", "target": "n1"})
    assert p.kind == "voltage" and p.target == "n1"


def test_ensure_probes_mixed() -> None:
    result = ensure_probes(["v(out)", {"kind": "current", "target": "R2"}, Probe.v("n2")])
    assert len(result) == 3
    kinds = {r.kind for r in result}
    assert kinds == {"voltage", "current"}
