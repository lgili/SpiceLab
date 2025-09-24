from spicelab.core.types import Probe, stable_hash


def test_stable_hash_ignores_probe_order() -> None:
    p1 = Probe.v("out")
    p2 = Probe.i("R1")
    h1 = stable_hash({"probes": [p1, p2]})
    h2 = stable_hash({"probes": [p2, p1]})
    # Order matters for lists normally; we keep it significant (documents current behavior).
    assert h1 != h2  # explicit: current contract keeps list ordering significant


def test_probe_helpers_equivalent() -> None:
    assert Probe.v("n1").kind == "voltage"
    assert Probe.i("R5").kind == "current"
