"""Property-based tests for Circuit class using Hypothesis.

These tests verify that circuits behave correctly across a wide range
of configurations, topologies, and component combinations.
"""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, invariant, rule
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Inductor, Resistor, Vdc
from spicelab.core.net import GND, Net

# ==============================================================================
# Hypothesis Strategies for Circuits
# ==============================================================================

# Valid circuit names (non-empty alphanumeric strings)
circuit_names = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc")),
    min_size=1,
    max_size=50,
).filter(lambda s: s and s[0].isalpha())

# Net names (alphanumeric, can start with number for node IDs)
net_names = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
    min_size=1,
    max_size=20,
)

# Component references
component_refs = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
    min_size=1,
    max_size=20,
).filter(lambda s: s and s[0].isalpha())

# Physical values
resistances = st.floats(min_value=1e-3, max_value=1e12, allow_nan=False, allow_infinity=False)
capacitances = st.floats(min_value=1e-15, max_value=1.0, allow_nan=False, allow_infinity=False)
inductances = st.floats(min_value=1e-12, max_value=1e3, allow_nan=False, allow_infinity=False)
voltages = st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False)


# ==============================================================================
# Basic Circuit Property Tests
# ==============================================================================


@pytest.mark.property
@given(name=circuit_names)
def test_circuit_creation_with_valid_name(name: str):
    """Circuit should accept any valid alphanumeric name."""
    circuit = Circuit(name)
    assert circuit.name == name
    assert len(circuit._components) == 0


@pytest.mark.property
@given(name=circuit_names, ref=component_refs, resistance=resistances)
def test_circuit_add_single_component(name: str, ref: str, resistance: float):
    """Circuit should accept adding a single component."""
    circuit = Circuit(name)
    r = Resistor(ref, resistance)

    circuit.add(r)

    assert len(circuit._components) == 1
    assert circuit._components[0] is r


@pytest.mark.property
@given(name=circuit_names, n_components=st.integers(min_value=1, max_value=20))
def test_circuit_add_multiple_components(name: str, n_components: int):
    """Circuit should correctly track multiple added components."""
    circuit = Circuit(name)

    components = []
    for i in range(n_components):
        r = Resistor(f"R{i}", 1000.0)
        circuit.add(r)
        components.append(r)

    assert len(circuit._components) == n_components
    for i, comp in enumerate(components):
        assert circuit._components[i] is comp


@pytest.mark.property
@given(name=circuit_names)
def test_circuit_add_returns_self(name: str):
    """Circuit.add() should return self for method chaining."""
    circuit = Circuit(name)
    r = Resistor("R1", 1000.0)

    result = circuit.add(r)

    assert result is circuit


@pytest.mark.property
@given(
    name=circuit_names,
    ref1=component_refs,
    ref2=component_refs,
    r1=resistances,
    r2=resistances,
)
def test_circuit_chained_add(name: str, ref1: str, ref2: str, r1: float, r2: float):
    """Circuit should support chained add() calls."""
    assume(ref1 != ref2)  # Unique references

    circuit = Circuit(name)
    res1 = Resistor(ref1, r1)
    res2 = Resistor(ref2, r2)

    circuit.add(res1).add(res2)

    assert len(circuit._components) == 2


# ==============================================================================
# Circuit Connection Property Tests
# ==============================================================================


@pytest.mark.property
@given(name=circuit_names, ref=component_refs, resistance=resistances)
def test_circuit_connect_to_gnd(name: str, ref: str, resistance: float):
    """Circuit should correctly connect ports to GND."""
    circuit = Circuit(name)
    r = Resistor(ref, resistance)
    circuit.add(r)

    circuit.connect(r.ports[0], GND)

    assert circuit._port_to_net.get(r.ports[0]) is GND


@pytest.mark.property
@given(name=circuit_names, ref=component_refs, resistance=resistances, net_name=net_names)
def test_circuit_connect_to_named_net(name: str, ref: str, resistance: float, net_name: str):
    """Circuit should correctly connect ports to named nets."""
    assume(net_name != "0")  # Reserved for GND

    circuit = Circuit(name)
    r = Resistor(ref, resistance)
    circuit.add(r)
    net = Net(net_name)

    circuit.connect(r.ports[0], net)

    assert circuit._port_to_net.get(r.ports[0]) is net


@pytest.mark.property
@given(
    name=circuit_names,
    ref1=component_refs,
    ref2=component_refs,
    r1=resistances,
    r2=resistances,
)
def test_circuit_connect_ports_together(name: str, ref1: str, ref2: str, r1: float, r2: float):
    """Circuit should correctly connect two ports together."""
    assume(ref1 != ref2)

    circuit = Circuit(name)
    res1 = Resistor(ref1, r1)
    res2 = Resistor(ref2, r2)
    circuit.add(res1, res2)

    # Connect port-to-port
    circuit.connect(res1.ports[1], res2.ports[0])

    # Both ports should share same net
    net1 = circuit._port_to_net.get(res1.ports[1])
    net2 = circuit._port_to_net.get(res2.ports[0])
    assert net1 is net2


@pytest.mark.property
@given(name=circuit_names)
def test_circuit_connect_returns_self(name: str):
    """Circuit.connect() should return self for method chaining."""
    circuit = Circuit(name)
    r = Resistor("R1", 1000.0)
    circuit.add(r)

    result = circuit.connect(r.ports[0], GND)

    assert result is circuit


# ==============================================================================
# Netlist Generation Property Tests
# ==============================================================================


@pytest.mark.property
@given(name=circuit_names, ref=component_refs, resistance=resistances)
def test_netlist_contains_circuit_name(name: str, ref: str, resistance: float):
    """Generated netlist should contain circuit name in title."""
    circuit = Circuit(name)
    r = Resistor(ref, resistance)
    circuit.add(r)
    circuit.connect(r.ports[0], GND)
    circuit.connect(r.ports[1], Net("out"))

    netlist = circuit.build_netlist()

    assert name in netlist


@pytest.mark.property
@given(name=circuit_names, ref=component_refs, resistance=resistances)
def test_netlist_contains_component_ref(name: str, ref: str, resistance: float):
    """Generated netlist should contain component references."""
    circuit = Circuit(name)
    r = Resistor(ref, resistance)
    circuit.add(r)
    circuit.connect(r.ports[0], GND)
    circuit.connect(r.ports[1], Net("out"))

    netlist = circuit.build_netlist()

    assert ref in netlist


@pytest.mark.property
@given(name=circuit_names, ref=component_refs, resistance=resistances)
def test_netlist_ends_with_dot_end(name: str, ref: str, resistance: float):
    """Generated netlist should end with .end directive."""
    circuit = Circuit(name)
    r = Resistor(ref, resistance)
    circuit.add(r)
    circuit.connect(r.ports[0], GND)
    circuit.connect(r.ports[1], Net("out"))

    netlist = circuit.build_netlist()

    assert ".end" in netlist.lower()


@pytest.mark.property
@given(name=circuit_names, n_components=st.integers(min_value=1, max_value=10))
def test_netlist_contains_all_components(name: str, n_components: int):
    """Generated netlist should contain all added components."""
    circuit = Circuit(name)

    refs = []
    prev_net = GND
    for i in range(n_components):
        ref = f"R{i}"
        refs.append(ref)
        r = Resistor(ref, 1000.0)
        circuit.add(r)
        circuit.connect(r.ports[0], prev_net)
        next_net = Net(f"n{i}")
        circuit.connect(r.ports[1], next_net)
        prev_net = next_net

    netlist = circuit.build_netlist()

    for ref in refs:
        assert ref in netlist


@pytest.mark.property
@given(name=circuit_names)
def test_netlist_is_deterministic(name: str):
    """Netlist generation should be deterministic for same circuit."""

    def build_test_circuit(circuit_name: str) -> Circuit:
        c = Circuit(circuit_name)
        r1 = Resistor("R1", 1000.0)
        r2 = Resistor("R2", 2000.0)
        c.add(r1, r2)
        c.connect(r1.ports[0], GND)
        c.connect(r1.ports[1], Net("mid"))
        c.connect(r2.ports[0], Net("mid"))
        c.connect(r2.ports[1], Net("out"))
        return c

    circuit1 = build_test_circuit(name)
    circuit2 = build_test_circuit(name)

    netlist1 = circuit1.build_netlist()
    netlist2 = circuit2.build_netlist()

    assert netlist1 == netlist2


# ==============================================================================
# Circuit Hash Property Tests
# ==============================================================================


@pytest.mark.property
@given(name=circuit_names, ref=component_refs, resistance=resistances)
def test_circuit_hash_is_string(name: str, ref: str, resistance: float):
    """Circuit hash should return a string."""
    circuit = Circuit(name)
    r = Resistor(ref, resistance)
    circuit.add(r)
    circuit.connect(r.ports[0], GND)
    circuit.connect(r.ports[1], Net("out"))

    h = circuit.hash()

    assert isinstance(h, str)
    assert len(h) > 0


@pytest.mark.property
@given(name=circuit_names)
def test_circuit_hash_is_deterministic(name: str):
    """Circuit hash should be deterministic for identical circuits."""

    def build_test_circuit(circuit_name: str) -> Circuit:
        c = Circuit(circuit_name)
        r = Resistor("R1", 1000.0)
        c.add(r)
        c.connect(r.ports[0], GND)
        c.connect(r.ports[1], Net("out"))
        return c

    circuit1 = build_test_circuit(name)
    circuit2 = build_test_circuit(name)

    assert circuit1.hash() == circuit2.hash()


@pytest.mark.property
@given(
    name=circuit_names,
    resistance1=resistances,
    resistance2=resistances,
)
def test_different_circuits_have_different_hashes(
    name: str, resistance1: float, resistance2: float
):
    """Different circuits should (usually) have different hashes."""
    assume(abs(resistance1 - resistance2) > 1e-6)  # Ensure values are different

    circuit1 = Circuit(name)
    r1 = Resistor("R1", resistance1)
    circuit1.add(r1)
    circuit1.connect(r1.ports[0], GND)
    circuit1.connect(r1.ports[1], Net("out"))

    circuit2 = Circuit(name)
    r2 = Resistor("R1", resistance2)
    circuit2.add(r2)
    circuit2.connect(r2.ports[0], GND)
    circuit2.connect(r2.ports[1], Net("out"))

    # Different resistance values should produce different hashes
    assert circuit1.hash() != circuit2.hash()


# ==============================================================================
# Circuit Summary Property Tests
# ==============================================================================


@pytest.mark.property
@given(name=circuit_names, ref=component_refs, resistance=resistances)
def test_circuit_summary_contains_name(name: str, ref: str, resistance: float):
    """Circuit summary should contain circuit name."""
    circuit = Circuit(name)
    r = Resistor(ref, resistance)
    circuit.add(r)
    circuit.connect(r.ports[0], GND)
    circuit.connect(r.ports[1], Net("out"))

    summary = circuit.summary()

    assert name in summary


@pytest.mark.property
@given(name=circuit_names, ref=component_refs, resistance=resistances)
def test_circuit_summary_contains_component_ref(name: str, ref: str, resistance: float):
    """Circuit summary should contain component references."""
    circuit = Circuit(name)
    r = Resistor(ref, resistance)
    circuit.add(r)
    circuit.connect(r.ports[0], GND)
    circuit.connect(r.ports[1], Net("out"))

    summary = circuit.summary()

    assert ref in summary


@pytest.mark.property
@given(name=circuit_names, n_components=st.integers(min_value=0, max_value=10))
def test_circuit_summary_contains_component_count(name: str, n_components: int):
    """Circuit summary should report correct component count."""
    circuit = Circuit(name)

    prev_net = GND
    for i in range(n_components):
        r = Resistor(f"R{i}", 1000.0)
        circuit.add(r)
        circuit.connect(r.ports[0], prev_net)
        next_net = Net(f"n{i}")
        circuit.connect(r.ports[1], next_net)
        prev_net = next_net

    summary = circuit.summary()

    assert f"Components ({n_components})" in summary


# ==============================================================================
# Directive Property Tests
# ==============================================================================


@pytest.mark.property
@given(name=circuit_names, directive=st.text(min_size=1, max_size=100))
def test_circuit_add_directive(name: str, directive: str):
    """Circuit should accept adding directives."""
    circuit = Circuit(name)

    circuit.add_directive(directive)

    assert len(circuit._directives) == 1
    assert circuit._directives[0] == directive.rstrip("\n")


@pytest.mark.property
@given(name=circuit_names)
def test_circuit_add_directive_returns_self(name: str):
    """Circuit.add_directive() should return self for chaining."""
    circuit = Circuit(name)

    result = circuit.add_directive(".model D1 D")

    assert result is circuit


@pytest.mark.property
@given(name=circuit_names, directive=st.text(min_size=1, max_size=100))
def test_circuit_add_directive_once_deduplicates(name: str, directive: str):
    """Circuit.add_directive_once() should not add duplicate directives."""
    circuit = Circuit(name)

    circuit.add_directive_once(directive)
    circuit.add_directive_once(directive)
    circuit.add_directive_once(f"  {directive}  ")  # With whitespace

    assert len(circuit._directives) == 1


@pytest.mark.property
@given(
    name=circuit_names,
    dir1=st.text(min_size=1, max_size=50),
    dir2=st.text(min_size=1, max_size=50),
)
def test_circuit_add_directive_once_allows_different(name: str, dir1: str, dir2: str):
    """Circuit.add_directive_once() should allow different directives."""
    assume(dir1.strip() != dir2.strip())

    circuit = Circuit(name)

    circuit.add_directive_once(dir1)
    circuit.add_directive_once(dir2)

    assert len(circuit._directives) == 2


# ==============================================================================
# Mixed Component Types Property Tests
# ==============================================================================


@pytest.mark.property
@given(
    name=circuit_names,
    r_val=resistances,
    c_val=capacitances,
    l_val=inductances,
    v_val=voltages,
)
def test_circuit_with_mixed_components(
    name: str, r_val: float, c_val: float, l_val: float, v_val: float
):
    """Circuit should handle mixed component types correctly."""
    circuit = Circuit(name)

    # Create components
    vin = Vdc("Vin", v_val)
    r = Resistor("R1", r_val)
    c = Capacitor("C1", c_val)
    ind = Inductor("L1", l_val)

    # Add all
    circuit.add(vin, r, c, ind)

    # Connect in series: Vin -> R -> C -> L -> GND
    circuit.connect(vin.ports[0], Net("in"))
    circuit.connect(vin.ports[1], GND)
    circuit.connect(r.ports[0], Net("in"))
    circuit.connect(r.ports[1], Net("n1"))
    circuit.connect(c.ports[0], Net("n1"))
    circuit.connect(c.ports[1], Net("n2"))
    circuit.connect(ind.ports[0], Net("n2"))
    circuit.connect(ind.ports[1], GND)

    # Should generate valid netlist
    netlist = circuit.build_netlist()

    assert "Vin" in netlist
    assert "R1" in netlist
    assert "C1" in netlist
    assert "L1" in netlist
    assert ".end" in netlist.lower()


# ==============================================================================
# Stateful Testing: Circuit Construction State Machine
# ==============================================================================


class CircuitStateMachine(RuleBasedStateMachine):
    """Stateful testing for Circuit operations.

    This state machine performs random valid operations on a circuit
    and verifies invariants are maintained throughout.
    """

    def __init__(self) -> None:
        super().__init__()
        self.circuit = Circuit("test_circuit")
        self.component_count = 0
        self.added_refs: set[str] = set()
        self.nets: list[Net] = [GND]

    # Bundles to track created objects
    components = Bundle("components")

    @rule(target=components, resistance=resistances)
    def add_resistor(self, resistance: float) -> Resistor:
        """Add a resistor with unique reference."""
        ref = f"R{self.component_count}"
        self.component_count += 1
        r = Resistor(ref, resistance)
        self.circuit.add(r)
        self.added_refs.add(ref)
        return r

    @rule(target=components, capacitance=capacitances)
    def add_capacitor(self, capacitance: float) -> Capacitor:
        """Add a capacitor with unique reference."""
        ref = f"C{self.component_count}"
        self.component_count += 1
        c = Capacitor(ref, capacitance)
        self.circuit.add(c)
        self.added_refs.add(ref)
        return c

    @rule(target=components, inductance=inductances)
    def add_inductor(self, inductance: float) -> Inductor:
        """Add an inductor with unique reference."""
        ref = f"L{self.component_count}"
        self.component_count += 1
        ind = Inductor(ref, inductance)
        self.circuit.add(ind)
        self.added_refs.add(ref)
        return ind

    @rule(target=components, voltage=voltages)
    def add_voltage_source(self, voltage: float) -> Vdc:
        """Add a voltage source with unique reference."""
        ref = f"V{self.component_count}"
        self.component_count += 1
        v = Vdc(ref, voltage)
        self.circuit.add(v)
        self.added_refs.add(ref)
        return v

    @rule(comp=components)
    def connect_to_gnd(self, comp: Resistor | Capacitor | Inductor | Vdc) -> None:
        """Connect a component's first port to ground."""
        self.circuit.connect(comp.ports[0], GND)

    @rule(comp=components, net_name=net_names)
    def connect_to_named_net(
        self, comp: Resistor | Capacitor | Inductor | Vdc, net_name: str
    ) -> None:
        """Connect a component's second port to a named net."""
        if net_name == "0":
            net_name = "n_alt"
        net = Net(net_name)
        self.circuit.connect(comp.ports[1], net)
        if net not in self.nets:
            self.nets.append(net)

    @rule()
    def add_directive(self) -> None:
        """Add a simple directive."""
        self.circuit.add_directive(".option RELTOL=1e-4")

    @invariant()
    def component_count_matches(self) -> None:
        """Invariant: component count should match what was added."""
        assert len(self.circuit._components) == len(self.added_refs)

    @invariant()
    def can_generate_summary(self) -> None:
        """Invariant: circuit should always be able to generate a summary."""
        summary = self.circuit.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    @invariant()
    def can_generate_netlist_if_connected(self) -> None:
        """Invariant: circuit should generate netlist if ports are connected."""
        # Only test if all ports are connected
        all_connected = True
        for comp in self.circuit._components:
            for port in comp.ports:
                if port not in self.circuit._port_to_net:
                    all_connected = False
                    break

        if all_connected and len(self.circuit._components) > 0:
            netlist = self.circuit.build_netlist()
            assert isinstance(netlist, str)
            assert ".end" in netlist.lower()


# Run the state machine test
TestCircuitStateMachine = CircuitStateMachine.TestCase
TestCircuitStateMachine.settings = settings(max_examples=50, stateful_step_count=20)
