from spicelab.core.types import AnalysisSpec, SweepSpec, ensure_analysis_spec, ensure_sweep_spec


def test_analysis_spec_valid() -> None:
    a = AnalysisSpec(mode="ac", args={"n": 10, "fstart": 1, "fstop": 10})
    assert a.mode == "ac"
    assert a.args["n"] == 10


def test_analysis_spec_reject_nested_obj() -> None:
    try:
        AnalysisSpec(mode="ac", args={"bad": object()})
    except Exception as e:  # noqa: BLE001
        assert "primitive" in str(e).lower()
    else:  # pragma: no cover
        raise AssertionError("Expected validation error")


def test_sweep_spec_normalizes_units() -> None:
    s = SweepSpec(variables={"R": ["1k", "2k", 3000]})
    vals = s.variables["R"]
    assert vals == [1000.0, 2000.0, 3000.0]


def test_normalizers_accept_dict() -> None:
    a = ensure_analysis_spec({"mode": "op", "args": {}})
    assert isinstance(a, AnalysisSpec)
    s = ensure_sweep_spec({"variables": {"X": [1, 2]}})
    assert isinstance(s, SweepSpec)


def test_analysis_spec_positional_constructor() -> None:
    a = AnalysisSpec("dc", {"src": "V1", "start": 0, "stop": 1, "step": 0.1})
    assert a.mode == "dc"
    assert a.args["src"] == "V1"
