"""Property-based tests for Net and Port classes using Hypothesis.

These tests verify that network topology primitives behave correctly
across a wide range of configurations.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from spicelab.core.components import Capacitor, Inductor, Resistor, Vdc
from spicelab.core.net import GND, Net, PortRole

# ==============================================================================
# Hypothesis Strategies
# ==============================================================================

# Valid net names
net_names = st.one_of(
    st.none(),
    st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc")),
        min_size=1,
        max_size=50,
    ),
)

# Port roles
port_roles = st.sampled_from([PortRole.POSITIVE, PortRole.NEGATIVE, PortRole.NODE])

# Port names
port_names = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
    min_size=1,
    max_size=20,
)


# ==============================================================================
# Net Property Tests
# ==============================================================================


@pytest.mark.property
@given(name=net_names)
def test_net_creation_with_name(name: str | None):
    """Net should accept any valid name including None."""
    net = Net(name)
    assert net.name == name


@pytest.mark.property
@given(name=net_names)
def test_net_is_frozen(name: str | None):
    """Net should be immutable (frozen dataclass)."""
    net = Net(name)

    with pytest.raises(AttributeError):
        net.name = "new_name"  # type: ignore[misc]


@pytest.mark.property
@given(name=net_names)
def test_net_equality_by_value(name: str | None):
    """Nets with same name should be equal."""
    net1 = Net(name)
    net2 = Net(name)

    assert net1 == net2


@pytest.mark.property
@given(name1=net_names, name2=net_names)
def test_net_inequality_by_value(name1: str | None, name2: str | None):
    """Nets with different names should not be equal."""
    if name1 != name2:
        net1 = Net(name1)
        net2 = Net(name2)
        assert net1 != net2


@pytest.mark.property
@given(name=net_names)
def test_net_is_hashable(name: str | None):
    """Net should be hashable for use in sets/dicts."""
    net = Net(name)

    # Should be usable as dict key
    d = {net: "value"}
    assert d[net] == "value"

    # Should be usable in set
    s = {net}
    assert net in s


@pytest.mark.property
@given(name=net_names)
def test_net_hash_is_deterministic(name: str | None):
    """Net hash should be deterministic for same name."""
    net1 = Net(name)
    net2 = Net(name)

    assert hash(net1) == hash(net2)


@pytest.mark.property
def test_gnd_is_singleton():
    """GND should be a predefined net with name '0'."""
    assert GND.name == "0"
    assert isinstance(GND, Net)


@pytest.mark.property
def test_gnd_equals_zero_net():
    """GND should equal Net('0')."""
    zero_net = Net("0")
    assert GND == zero_net


# ==============================================================================
# Port Property Tests
# ==============================================================================


@pytest.mark.property
@given(resistance=st.floats(min_value=1, max_value=1e6, allow_nan=False))
def test_port_has_owner(resistance: float):
    """Port should correctly reference its owning component."""
    r = Resistor("R1", resistance)

    for port in r.ports:
        assert port.owner is r


@pytest.mark.property
@given(resistance=st.floats(min_value=1, max_value=1e6, allow_nan=False))
def test_port_has_name(resistance: float):
    """Port should have a name."""
    r = Resistor("R1", resistance)

    for port in r.ports:
        assert isinstance(port.name, str)
        assert len(port.name) > 0


@pytest.mark.property
@given(resistance=st.floats(min_value=1, max_value=1e6, allow_nan=False))
def test_port_has_role(resistance: float):
    """Port should have a PortRole."""
    r = Resistor("R1", resistance)

    for port in r.ports:
        assert isinstance(port.role, PortRole)


@pytest.mark.property
@given(resistance=st.floats(min_value=1, max_value=1e6, allow_nan=False))
def test_port_is_frozen(resistance: float):
    """Port should be immutable (frozen dataclass)."""
    r = Resistor("R1", resistance)
    port = r.ports[0]

    with pytest.raises(AttributeError):
        port.name = "new_name"  # type: ignore[misc]


@pytest.mark.property
@given(resistance=st.floats(min_value=1, max_value=1e6, allow_nan=False))
def test_port_identity_based_equality(resistance: float):
    """Port should use identity-based equality (eq=False)."""
    r1 = Resistor("R1", resistance)
    r2 = Resistor("R1", resistance)  # Same params

    # Different resistor instances should have different ports
    assert r1.ports[0] is not r2.ports[0]
    assert r1.ports[0] != r2.ports[0]  # Identity-based, not value-based


@pytest.mark.property
@given(resistance=st.floats(min_value=1, max_value=1e6, allow_nan=False))
def test_port_is_hashable(resistance: float):
    """Port should be hashable by identity for use in dicts."""
    r = Resistor("R1", resistance)
    port = r.ports[0]

    # Should be usable as dict key
    d = {port: "connected"}
    assert d[port] == "connected"

    # Should be usable in set
    s = {port}
    assert port in s


# ==============================================================================
# PortRole Property Tests
# ==============================================================================


@pytest.mark.property
@given(role=port_roles)
def test_port_role_is_enum(role: PortRole):
    """PortRole should be an Enum value."""
    assert isinstance(role, PortRole)


@pytest.mark.property
def test_port_role_has_expected_values():
    """PortRole should have POSITIVE, NEGATIVE, and NODE values."""
    assert hasattr(PortRole, "POSITIVE")
    assert hasattr(PortRole, "NEGATIVE")
    assert hasattr(PortRole, "NODE")


@pytest.mark.property
def test_port_roles_are_distinct():
    """All PortRole values should be distinct."""
    roles = [PortRole.POSITIVE, PortRole.NEGATIVE, PortRole.NODE]
    assert len(set(roles)) == 3


# ==============================================================================
# Component Port Structure Property Tests
# ==============================================================================


@pytest.mark.property
@given(resistance=st.floats(min_value=1, max_value=1e6, allow_nan=False))
def test_resistor_has_two_ports(resistance: float):
    """Resistor should have exactly 2 ports."""
    r = Resistor("R1", resistance)
    assert len(r.ports) == 2


@pytest.mark.property
@given(capacitance=st.floats(min_value=1e-15, max_value=1, allow_nan=False))
def test_capacitor_has_two_ports(capacitance: float):
    """Capacitor should have exactly 2 ports."""
    c = Capacitor("C1", capacitance)
    assert len(c.ports) == 2


@pytest.mark.property
@given(inductance=st.floats(min_value=1e-12, max_value=1e3, allow_nan=False))
def test_inductor_has_two_ports(inductance: float):
    """Inductor should have exactly 2 ports."""
    ind = Inductor("L1", inductance)
    assert len(ind.ports) == 2


@pytest.mark.property
@given(voltage=st.floats(min_value=-1000, max_value=1000, allow_nan=False))
def test_vdc_has_two_ports(voltage: float):
    """Vdc should have exactly 2 ports."""
    v = Vdc("V1", voltage)
    assert len(v.ports) == 2


@pytest.mark.property
@given(resistance=st.floats(min_value=1, max_value=1e6, allow_nan=False))
def test_component_ports_are_unique(resistance: float):
    """Component ports should be unique objects."""
    r = Resistor("R1", resistance)

    # Ports should be different objects
    assert r.ports[0] is not r.ports[1]


@pytest.mark.property
@given(resistance=st.floats(min_value=1, max_value=1e6, allow_nan=False))
def test_component_ports_have_distinct_names(resistance: float):
    """Component ports should have distinct names."""
    r = Resistor("R1", resistance)

    names = [p.name for p in r.ports]
    assert len(names) == len(set(names))


# ==============================================================================
# Net/Port Interaction Property Tests
# ==============================================================================


@pytest.mark.property
@given(
    net_name=st.text(min_size=1, max_size=20).filter(lambda s: s != "0"),
    resistance=st.floats(min_value=1, max_value=1e6, allow_nan=False),
)
def test_net_can_be_used_with_multiple_ports(net_name: str, resistance: float):
    """Same Net can be connected to multiple ports in a circuit."""
    from spicelab.core.circuit import Circuit

    circuit = Circuit("test")
    r1 = Resistor("R1", resistance)
    r2 = Resistor("R2", resistance)
    circuit.add(r1, r2)

    shared_net = Net(net_name)
    circuit.connect(r1.ports[1], shared_net)
    circuit.connect(r2.ports[0], shared_net)

    # Both ports should be connected to same net
    assert circuit._port_to_net.get(r1.ports[1]) is shared_net
    assert circuit._port_to_net.get(r2.ports[0]) is shared_net


@pytest.mark.property
@given(resistance=st.floats(min_value=1, max_value=1e6, allow_nan=False))
def test_gnd_can_be_connected_to_multiple_ports(resistance: float):
    """GND can be connected to multiple ports."""
    from spicelab.core.circuit import Circuit

    circuit = Circuit("test")
    r1 = Resistor("R1", resistance)
    r2 = Resistor("R2", resistance)
    circuit.add(r1, r2)

    circuit.connect(r1.ports[0], GND)
    circuit.connect(r2.ports[0], GND)

    # Both should reference GND
    assert circuit._port_to_net.get(r1.ports[0]) is GND
    assert circuit._port_to_net.get(r2.ports[0]) is GND


# ==============================================================================
# Edge Case Property Tests
# ==============================================================================


@pytest.mark.property
def test_net_with_empty_string_name():
    """Net should accept empty string as name."""
    net = Net("")
    assert net.name == ""


@pytest.mark.property
@given(
    name=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc", "Zs")),
        min_size=1,
        max_size=100,
    )
)
def test_net_with_various_characters(name: str):
    """Net should accept names with various character types."""
    net = Net(name)
    assert net.name == name


@pytest.mark.property
def test_net_with_numeric_name():
    """Net should accept numeric string names."""
    net = Net("123")
    assert net.name == "123"


@pytest.mark.property
def test_net_with_reserved_zero_name():
    """Net('0') should be valid and equal to GND."""
    net = Net("0")
    assert net.name == "0"
    assert net == GND
