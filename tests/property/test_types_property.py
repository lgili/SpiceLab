"""Property-based tests for core types using Hypothesis.

These tests verify that AnalysisSpec, SweepSpec, Probe, and hash utilities
behave correctly across a wide range of inputs.
"""

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from spicelab.core.types import (
    AnalysisSpec,
    Probe,
    SweepSpec,
    circuit_hash,
    ensure_analysis_spec,
    ensure_probe,
    ensure_probes,
    ensure_sweep_spec,
    stable_hash,
)

# ==============================================================================
# Hypothesis Strategies for Types
# ==============================================================================

# Valid analysis modes
analysis_modes = st.sampled_from(["op", "dc", "ac", "tran", "noise"])

# Valid probe kinds
probe_kinds = st.sampled_from(["voltage", "current"])

# Non-empty strings for targets/names
non_empty_strings = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc")),
    min_size=1,
    max_size=50,
)

# Primitive values for args
primitive_values = st.one_of(
    st.integers(min_value=-10000, max_value=10000),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.text(min_size=0, max_size=50),
)

# Simple arg dictionaries (valid for AnalysisSpec)
simple_args = st.dictionaries(
    keys=non_empty_strings,
    values=primitive_values,
    max_size=10,
)

# Float values for sweeps
sweep_values = st.floats(min_value=1e-12, max_value=1e12, allow_nan=False, allow_infinity=False)


# ==============================================================================
# AnalysisSpec Property Tests
# ==============================================================================


@pytest.mark.property
@given(mode=analysis_modes)
def test_analysis_spec_accepts_valid_modes(mode: str):
    """AnalysisSpec should accept all valid analysis modes."""
    spec = AnalysisSpec(mode=mode)  # type: ignore[arg-type]
    assert spec.mode == mode


@pytest.mark.property
@given(mode=analysis_modes, args=simple_args)
def test_analysis_spec_accepts_simple_args(mode: str, args: dict):
    """AnalysisSpec should accept dictionaries with primitive values."""
    spec = AnalysisSpec(mode=mode, args=args)  # type: ignore[arg-type]
    assert spec.mode == mode
    assert spec.args == args


@pytest.mark.property
@given(mode=analysis_modes)
def test_analysis_spec_default_args_empty(mode: str):
    """AnalysisSpec should have empty args by default."""
    spec = AnalysisSpec(mode=mode)  # type: ignore[arg-type]
    assert spec.args == {}


@pytest.mark.property
@given(mode=analysis_modes)
def test_analysis_spec_is_hashable_via_model_dump(mode: str):
    """AnalysisSpec should be serializable via model_dump."""
    spec = AnalysisSpec(mode=mode)  # type: ignore[arg-type]
    dumped = spec.model_dump()

    assert isinstance(dumped, dict)
    assert dumped["mode"] == mode


@pytest.mark.property
@given(mode=analysis_modes, args=simple_args)
def test_analysis_spec_roundtrip(mode: str, args: dict):
    """AnalysisSpec should survive dump/load roundtrip."""
    spec = AnalysisSpec(mode=mode, args=args)  # type: ignore[arg-type]
    dumped = spec.model_dump()

    # Recreate from dict
    recreated = AnalysisSpec(**dumped)

    assert recreated.mode == spec.mode
    assert recreated.args == spec.args


# ==============================================================================
# SweepSpec Property Tests
# ==============================================================================


@pytest.mark.property
@given(var_name=non_empty_strings, values=st.lists(sweep_values, min_size=1, max_size=20))
def test_sweep_spec_accepts_float_values(var_name: str, values: list):
    """SweepSpec should accept lists of float values."""
    spec = SweepSpec(variables={var_name: values})
    assert var_name in spec.variables
    # All values should be normalized to floats
    assert all(isinstance(v, float) for v in spec.variables[var_name])


@pytest.mark.property
@given(
    var_name=non_empty_strings,
    values=st.lists(
        st.sampled_from(["1k", "10k", "100k", "1M", "1m", "1u", "1n", "1p"]),
        min_size=1,
        max_size=10,
    ),
)
def test_sweep_spec_normalizes_unit_strings(var_name: str, values: list):
    """SweepSpec should normalize string values with SI prefixes to floats."""
    spec = SweepSpec(variables={var_name: values})

    # All values should be converted to floats
    assert all(isinstance(v, float) for v in spec.variables[var_name])


@pytest.mark.property
@given(
    var1=non_empty_strings,
    var2=non_empty_strings,
    vals1=st.lists(sweep_values, min_size=1, max_size=5),
    vals2=st.lists(sweep_values, min_size=1, max_size=5),
)
def test_sweep_spec_accepts_multiple_variables(var1: str, var2: str, vals1: list, vals2: list):
    """SweepSpec should accept multiple sweep variables."""
    assume(var1 != var2)

    spec = SweepSpec(variables={var1: vals1, var2: vals2})

    assert var1 in spec.variables
    assert var2 in spec.variables


@pytest.mark.property
@given(var_name=non_empty_strings)
def test_sweep_spec_rejects_empty_values(var_name: str):
    """SweepSpec should reject empty value lists."""
    with pytest.raises(ValueError, match="empty"):
        SweepSpec(variables={var_name: []})


# ==============================================================================
# Probe Property Tests
# ==============================================================================


@pytest.mark.property
@given(kind=probe_kinds, target=non_empty_strings)
def test_probe_accepts_valid_inputs(kind: str, target: str):
    """Probe should accept valid kind and target combinations."""
    probe = Probe(kind=kind, target=target)  # type: ignore[arg-type]
    assert probe.kind == kind
    assert probe.target == target


@pytest.mark.property
@given(target=non_empty_strings)
def test_probe_v_shortcut(target: str):
    """Probe.v() should create a voltage probe."""
    probe = Probe.v(target)
    assert probe.kind == "voltage"
    assert probe.target == target


@pytest.mark.property
@given(target=non_empty_strings)
def test_probe_i_shortcut(target: str):
    """Probe.i() should create a current probe."""
    probe = Probe.i(target)
    assert probe.kind == "current"
    assert probe.target == target


@pytest.mark.property
@given(kind=probe_kinds, target=non_empty_strings)
def test_probe_roundtrip(kind: str, target: str):
    """Probe should survive dump/load roundtrip."""
    probe = Probe(kind=kind, target=target)  # type: ignore[arg-type]
    dumped = probe.model_dump()

    recreated = Probe(**dumped)

    assert recreated.kind == probe.kind
    assert recreated.target == probe.target


@pytest.mark.property
@given(kind=probe_kinds)
def test_probe_rejects_empty_target(kind: str):
    """Probe should reject empty target strings."""
    with pytest.raises(ValueError, match="non-empty"):
        Probe(kind=kind, target="")  # type: ignore[arg-type]


@pytest.mark.property
@given(kind=probe_kinds)
def test_probe_rejects_whitespace_target(kind: str):
    """Probe should reject whitespace-only target strings."""
    with pytest.raises(ValueError, match="non-empty"):
        Probe(kind=kind, target="   ")  # type: ignore[arg-type]


# ==============================================================================
# ensure_* Function Property Tests
# ==============================================================================


@pytest.mark.property
@given(mode=analysis_modes, args=simple_args)
def test_ensure_analysis_spec_passthrough(mode: str, args: dict):
    """ensure_analysis_spec should pass through AnalysisSpec instances."""
    spec = AnalysisSpec(mode=mode, args=args)  # type: ignore[arg-type]
    result = ensure_analysis_spec(spec)
    assert result is spec


@pytest.mark.property
@given(mode=analysis_modes, args=simple_args)
def test_ensure_analysis_spec_from_dict(mode: str, args: dict):
    """ensure_analysis_spec should create spec from valid dict."""
    d = {"mode": mode, "args": args}
    result = ensure_analysis_spec(d)

    assert isinstance(result, AnalysisSpec)
    assert result.mode == mode
    assert result.args == args


@pytest.mark.property
@given(
    mode=st.text(min_size=1, max_size=20).filter(
        lambda s: s not in ["op", "dc", "ac", "tran", "noise"]
    )
)
def test_ensure_analysis_spec_rejects_invalid_mode(mode: str):
    """ensure_analysis_spec should reject invalid modes."""
    with pytest.raises(ValueError, match="Invalid analysis mode"):
        ensure_analysis_spec({"mode": mode})


@pytest.mark.property
@given(kind=probe_kinds, target=non_empty_strings)
def test_ensure_probe_passthrough(kind: str, target: str):
    """ensure_probe should pass through Probe instances."""
    probe = Probe(kind=kind, target=target)  # type: ignore[arg-type]
    result = ensure_probe(probe)
    assert result is probe


@pytest.mark.property
@given(kind=probe_kinds, target=non_empty_strings)
def test_ensure_probe_from_dict(kind: str, target: str):
    """ensure_probe should create probe from valid dict."""
    d = {"kind": kind, "target": target}
    result = ensure_probe(d)

    assert isinstance(result, Probe)
    assert result.kind == kind
    assert result.target == target


@pytest.mark.property
@given(target=non_empty_strings)
def test_ensure_probe_from_v_string(target: str):
    """ensure_probe should parse V(node) string format."""
    result = ensure_probe(f"V({target})")

    assert result.kind == "voltage"
    assert result.target == target


@pytest.mark.property
@given(target=non_empty_strings)
def test_ensure_probe_from_i_string(target: str):
    """ensure_probe should parse I(ref) string format."""
    result = ensure_probe(f"I({target})")

    assert result.kind == "current"
    assert result.target == target


@pytest.mark.property
@given(target=non_empty_strings)
def test_ensure_probe_bare_string_is_voltage(target: str):
    """ensure_probe should treat bare strings as voltage probes."""
    result = ensure_probe(target)

    assert result.kind == "voltage"
    assert result.target == target


@pytest.mark.property
@given(probes=st.lists(non_empty_strings, min_size=0, max_size=10))
def test_ensure_probes_converts_all(probes: list):
    """ensure_probes should convert all items in list."""
    result = ensure_probes(probes)

    assert len(result) == len(probes)
    assert all(isinstance(p, Probe) for p in result)


@pytest.mark.property
def test_ensure_probes_empty_list():
    """ensure_probes should return empty list for None or empty input."""
    assert ensure_probes(None) == []
    assert ensure_probes([]) == []


@pytest.mark.property
@given(
    var_name=non_empty_strings,
    values=st.lists(sweep_values, min_size=1, max_size=5),
)
def test_ensure_sweep_spec_passthrough(var_name: str, values: list):
    """ensure_sweep_spec should pass through SweepSpec instances."""
    spec = SweepSpec(variables={var_name: values})
    result = ensure_sweep_spec(spec)
    assert result is spec


@pytest.mark.property
@given(
    var_name=non_empty_strings,
    values=st.lists(sweep_values, min_size=1, max_size=5),
)
def test_ensure_sweep_spec_from_dict(var_name: str, values: list):
    """ensure_sweep_spec should create spec from valid dict."""
    d = {"variables": {var_name: values}}
    result = ensure_sweep_spec(d)

    assert isinstance(result, SweepSpec)
    assert var_name in result.variables


@pytest.mark.property
def test_ensure_sweep_spec_none_returns_none():
    """ensure_sweep_spec should return None for None input."""
    assert ensure_sweep_spec(None) is None


# ==============================================================================
# Hash Function Property Tests
# ==============================================================================


@pytest.mark.property
@given(data=st.one_of(primitive_values, st.lists(primitive_values, max_size=10)))
def test_stable_hash_returns_string(data):
    """stable_hash should return a string hash."""
    h = stable_hash(data)

    assert isinstance(h, str)
    assert len(h) == 12  # SHA1 hex truncated to 12 chars


@pytest.mark.property
@given(data=st.one_of(primitive_values, st.lists(primitive_values, max_size=5)))
def test_stable_hash_is_deterministic(data):
    """stable_hash should be deterministic for same input."""
    h1 = stable_hash(data)
    h2 = stable_hash(data)

    assert h1 == h2


@pytest.mark.property
@given(
    data1=st.integers(min_value=0, max_value=1000),
    data2=st.integers(min_value=0, max_value=1000),
)
def test_stable_hash_different_for_different_inputs(data1: int, data2: int):
    """stable_hash should (usually) produce different hashes for different inputs."""
    assume(data1 != data2)

    h1 = stable_hash(data1)
    h2 = stable_hash(data2)

    assert h1 != h2


@pytest.mark.property
@given(
    d=st.dictionaries(
        keys=st.text(min_size=1, max_size=10),
        values=primitive_values,
        min_size=1,
        max_size=5,
    )
)
def test_stable_hash_dict_order_independent(d: dict):
    """stable_hash should produce same hash regardless of dict key order."""
    # Create a dict with keys in different order
    reversed_items = list(reversed(list(d.items())))
    d2 = dict(reversed_items)

    # Should have same hash (both dicts have same key-value pairs)
    h1 = stable_hash(d)
    h2 = stable_hash(d2)

    assert h1 == h2


@pytest.mark.property
@given(
    mode=analysis_modes,
    args=simple_args,
)
def test_stable_hash_works_with_pydantic_models(mode: str, args: dict):
    """stable_hash should correctly hash Pydantic models."""
    spec = AnalysisSpec(mode=mode, args=args)  # type: ignore[arg-type]

    h = stable_hash(spec)

    assert isinstance(h, str)
    assert len(h) == 12


@pytest.mark.property
@given(extra=st.dictionaries(non_empty_strings, primitive_values, max_size=5))
def test_circuit_hash_with_extra_context(extra: dict):
    """circuit_hash should incorporate extra context into hash."""
    from spicelab.core.circuit import Circuit
    from spicelab.core.components import Resistor
    from spicelab.core.net import GND, Net

    circuit = Circuit("test")
    r = Resistor("R1", 1000.0)
    circuit.add(r)
    circuit.connect(r.ports[0], GND)
    circuit.connect(r.ports[1], Net("out"))

    h1 = circuit_hash(circuit)
    h2 = circuit_hash(circuit, extra=extra)

    if extra:
        # Extra context should change hash
        assert h1 != h2
    else:
        # Empty extra should not change hash
        assert h1 == h2


@pytest.mark.property
def test_circuit_hash_is_deterministic():
    """circuit_hash should be deterministic for identical circuits."""
    from spicelab.core.circuit import Circuit
    from spicelab.core.components import Resistor
    from spicelab.core.net import GND, Net

    def build_circuit() -> Circuit:
        c = Circuit("test")
        r = Resistor("R1", 1000.0)
        c.add(r)
        c.connect(r.ports[0], GND)
        c.connect(r.ports[1], Net("out"))
        return c

    c1 = build_circuit()
    c2 = build_circuit()

    assert circuit_hash(c1) == circuit_hash(c2)


# ==============================================================================
# Edge Case Property Tests
# ==============================================================================


@pytest.mark.property
@given(mode=analysis_modes)
def test_analysis_spec_with_list_args(mode: str):
    """AnalysisSpec should accept lists of primitives in args."""
    args = {"values": [1, 2, 3], "names": ["a", "b", "c"]}
    spec = AnalysisSpec(mode=mode, args=args)  # type: ignore[arg-type]

    assert spec.args["values"] == [1, 2, 3]
    assert spec.args["names"] == ["a", "b", "c"]


@pytest.mark.property
@given(mode=analysis_modes)
def test_analysis_spec_with_tuple_args(mode: str):
    """AnalysisSpec should accept tuples of primitives in args."""
    args = {"values": (1.0, 2.0, 3.0)}
    spec = AnalysisSpec(mode=mode, args=args)  # type: ignore[arg-type]

    assert spec.args["values"] == (1.0, 2.0, 3.0)


@pytest.mark.property
@given(mode=analysis_modes)
def test_analysis_spec_with_none_args(mode: str):
    """AnalysisSpec should accept None values in args."""
    args = {"optional_param": None}
    spec = AnalysisSpec(mode=mode, args=args)  # type: ignore[arg-type]

    assert spec.args["optional_param"] is None


@pytest.mark.property
@given(target=non_empty_strings)
def test_ensure_probe_case_insensitive(target: str):
    """ensure_probe should handle case-insensitive V() and I() prefixes."""
    # Uppercase
    v_upper = ensure_probe(f"V({target})")
    assert v_upper.kind == "voltage"

    # Lowercase
    v_lower = ensure_probe(f"v({target})")
    assert v_lower.kind == "voltage"

    # Mixed case for current
    i_upper = ensure_probe(f"I({target})")
    assert i_upper.kind == "current"

    i_lower = ensure_probe(f"i({target})")
    assert i_lower.kind == "current"
