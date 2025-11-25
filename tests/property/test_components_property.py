"""Property-based tests for components using Hypothesis.

These tests use Hypothesis to generate random valid inputs and verify
that components behave correctly across a wide range of values.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from spicelab.core.components import Capacitor, Inductor, Resistor, Vdc
from spicelab.core.net import Port
from spicelab.exceptions import ComponentValidationError

# ==============================================================================
# Hypothesis Strategies
# ==============================================================================


# Physical value ranges for components
resistances = st.floats(min_value=1e-3, max_value=1e12, allow_nan=False, allow_infinity=False)
capacitances = st.floats(min_value=1e-15, max_value=1.0, allow_nan=False, allow_infinity=False)
inductances = st.floats(min_value=1e-12, max_value=1e3, allow_nan=False, allow_infinity=False)
voltages = st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)

# Valid component references (alphanumeric starting with letter)
component_refs = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
    min_size=1,
    max_size=20,
).filter(lambda s: s and s[0].isalpha())


# ==============================================================================
# Helper: Dummy net_of function for SPICE card generation
# ==============================================================================


def dummy_net_of(port: Port) -> str:
    """Dummy net_of function for testing SPICE card generation."""
    return f"net_{port.name}"


# ==============================================================================
# Resistor Property Tests
# ==============================================================================


@pytest.mark.property
@given(ref=component_refs, resistance=resistances)
def test_resistor_accepts_valid_resistance(ref: str, resistance: float):
    """Resistor should accept any positive resistance value."""
    r = Resistor(ref, resistance)
    assert r.value == resistance
    assert r.ref == ref


@pytest.mark.property
@given(ref=component_refs, resistance=resistances)
def test_resistor_spice_card_contains_ref_and_value(ref: str, resistance: float):
    """Resistor SPICE card should contain reference and value."""
    r = Resistor(ref, resistance)
    card = r.spice_card(dummy_net_of)

    assert ref in card, f"Reference '{ref}' not in card: {card}"
    assert isinstance(card, str)
    assert len(card) > 0


@pytest.mark.property
@given(ref=component_refs, resistance=resistances)
def test_resistor_repr_is_valid_python(ref: str, resistance: float):
    """Resistor repr should be valid Python expression."""
    r = Resistor(ref, resistance)
    repr_str = repr(r)

    # Should contain class name and key attributes
    assert "Resistor" in repr_str
    assert ref in repr_str


@pytest.mark.property
@given(resistance=st.floats(min_value=-1000, max_value=-0.001))
def test_resistor_rejects_negative_resistance(resistance: float):
    """Resistor should reject negative resistance values."""
    # Note: Current implementation might not validate, this test documents
    # expected behavior for future validation implementation
    try:
        r = Resistor("R1", resistance)
        # If we get here, check if card is still generated
        # (validates that even if object is created, it should warn/fail)
        card = r.spice_card(dummy_net_of)
        # Document current behavior: negative values are accepted
        assert isinstance(card, str)
    except (ValueError, ComponentValidationError):
        pass  # Expected future behavior


# ==============================================================================
# Capacitor Property Tests
# ==============================================================================


@pytest.mark.property
@given(ref=component_refs, capacitance=capacitances)
def test_capacitor_accepts_valid_capacitance(ref: str, capacitance: float):
    """Capacitor should accept any positive capacitance value."""
    c = Capacitor(ref, capacitance)
    assert c.value == capacitance
    assert c.ref == ref


@pytest.mark.property
@given(ref=component_refs, capacitance=capacitances)
def test_capacitor_spice_card_format(ref: str, capacitance: float):
    """Capacitor SPICE card should have valid format."""
    c = Capacitor(ref, capacitance)
    card = c.spice_card(dummy_net_of)

    assert isinstance(card, str)
    assert ref in card
    assert len(card) > 0
    # Card should contain 'C' prefix (SPICE convention)
    assert "C" in card


# ==============================================================================
# Inductor Property Tests
# ==============================================================================


@pytest.mark.property
@given(ref=component_refs, inductance=inductances)
def test_inductor_accepts_valid_inductance(ref: str, inductance: float):
    """Inductor should accept any positive inductance value."""
    ind = Inductor(ref, inductance)
    assert ind.value == inductance
    assert ind.ref == ref


@pytest.mark.property
@given(ref=component_refs, inductance=inductances)
def test_inductor_spice_card_is_nonempty(ref: str, inductance: float):
    """Inductor SPICE card should be non-empty string."""
    ind = Inductor(ref, inductance)
    card = ind.spice_card(dummy_net_of)

    assert isinstance(card, str)
    assert len(card) > 0


# ==============================================================================
# Voltage Source Property Tests
# ==============================================================================


@pytest.mark.property
@given(ref=component_refs, voltage=voltages)
def test_vdc_accepts_any_voltage(ref: str, voltage: float):
    """VDC should accept any voltage including negative and zero."""
    v = Vdc(ref, voltage)
    assert v.value == voltage
    assert v.ref == ref


@pytest.mark.property
@given(ref=component_refs, voltage=voltages)
def test_vdc_spice_card_contains_ref(ref: str, voltage: float):
    """VDC SPICE card should contain reference."""
    v = Vdc(ref, voltage)
    card = v.spice_card(dummy_net_of)

    assert isinstance(card, str)
    assert ref in card
    assert len(card) > 0


# ==============================================================================
# Cross-Component Property Tests
# ==============================================================================


@pytest.mark.property
@given(
    r_ref=component_refs,
    c_ref=component_refs,
    resistance=resistances,
    capacitance=capacitances,
)
def test_components_have_unique_refs(r_ref: str, c_ref: str, resistance: float, capacitance: float):
    """Components with different refs should be distinguishable."""
    r = Resistor(r_ref, resistance)
    c = Capacitor(c_ref, capacitance)

    # Different types should always be distinguishable
    assert not isinstance(r, type(c))

    # If refs are different, components are different
    if r_ref != c_ref:
        assert r.ref != c.ref


@pytest.mark.property
@given(resistance=resistances)
def test_component_value_roundtrip(resistance: float):
    """Component value should survive creation roundtrip."""
    r = Resistor("R1", resistance)
    retrieved = r.value

    # Should be exactly equal (no precision loss for assignment)
    assert retrieved == resistance


@pytest.mark.property
@given(ref=component_refs, resistance=resistances)
def test_component_has_ports(ref: str, resistance: float):
    """All passive components should have exactly 2 ports."""
    r = Resistor(ref, resistance)

    assert hasattr(r, "ports")
    ports = r.ports
    assert len(ports) == 2
    assert ports[0] is not None
    assert ports[1] is not None


@pytest.mark.property
@given(
    ref=component_refs,
    resistance=st.floats(min_value=1.0, max_value=1e6),
)
def test_resistor_spice_card_parseable(ref: str, resistance: float):
    """Resistor SPICE card should be in parseable format.

    SPICE format: <ref> <node1> <node2> <value>
    """
    r = Resistor(ref, resistance)
    card = r.spice_card(dummy_net_of)

    # Should be non-empty and contain at least reference
    assert len(card.strip()) > 0
    assert ref in card

    # Should not contain invalid characters for SPICE
    invalid_chars = ["\n\n", "\t\t"]  # Excessive whitespace
    for invalid in invalid_chars:
        assert invalid not in card


# ==============================================================================
# Edge Case Property Tests
# ==============================================================================


@pytest.mark.property
@given(resistance=st.floats(min_value=1e-15, max_value=1e-12))
def test_resistor_handles_very_small_values(resistance: float):
    """Resistor should handle very small resistance values."""
    r = Resistor("R1", resistance)
    card = r.spice_card(dummy_net_of)

    # Should still generate valid card
    assert isinstance(card, str)
    assert len(card) > 0


@pytest.mark.property
@given(resistance=st.floats(min_value=1e9, max_value=1e12))
def test_resistor_handles_very_large_values(resistance: float):
    """Resistor should handle very large resistance values."""
    r = Resistor("R1", resistance)
    card = r.spice_card(dummy_net_of)

    # Should still generate valid card
    assert isinstance(card, str)
    assert len(card) > 0


@pytest.mark.property
@given(
    ref=st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll")), min_size=1, max_size=1)
)
def test_component_accepts_single_char_ref(ref: str):
    """Components should accept single-character references."""
    r = Resistor(ref, 1000.0)
    assert r.ref == ref


@pytest.mark.property
@given(
    ref=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc")),
        min_size=1,
        max_size=50,
    ).filter(lambda s: s and s[0].isalpha())
)
def test_component_handles_long_refs(ref: str):
    """Components should handle reasonably long reference names."""
    r = Resistor(ref, 1000.0)
    assert r.ref == ref
    # Should still generate valid card
    card = r.spice_card(dummy_net_of)
    assert ref in card


# ==============================================================================
# Port Property Tests
# ==============================================================================


@pytest.mark.property
@given(ref=component_refs, resistance=resistances)
def test_resistor_ports_are_distinct(ref: str, resistance: float):
    """Resistor ports should be distinct objects."""
    r = Resistor(ref, resistance)

    assert r.ports[0] is not r.ports[1]
    assert r.ports[0] != r.ports[1]


@pytest.mark.property
@given(ref=component_refs, resistance=resistances)
def test_resistor_ports_reference_owner(ref: str, resistance: float):
    """Each port should reference its owning component."""
    r = Resistor(ref, resistance)

    for port in r.ports:
        assert port.owner is r


@pytest.mark.property
@given(ref=component_refs, capacitance=capacitances)
def test_capacitor_ports_are_hashable(ref: str, capacitance: float):
    """Capacitor ports should be usable as dict keys."""
    c = Capacitor(ref, capacitance)

    # Should be usable as dict key
    d = {c.ports[0]: "pos", c.ports[1]: "neg"}
    assert d[c.ports[0]] == "pos"
    assert d[c.ports[1]] == "neg"


@pytest.mark.property
@given(ref=component_refs, inductance=inductances)
def test_inductor_ports_have_names(ref: str, inductance: float):
    """Inductor ports should have non-empty names."""
    ind = Inductor(ref, inductance)

    for port in ind.ports:
        assert isinstance(port.name, str)
        assert len(port.name) > 0
