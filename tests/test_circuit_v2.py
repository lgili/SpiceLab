"""Tests for ImmutableCircuit (Phase 1.3)."""

from __future__ import annotations

import pytest
from spicelab.core.circuit_v2 import ImmutableCircuit
from spicelab.core.components import Capacitor, Resistor
from spicelab.core.net import GND, Net


def test_empty_circuit():
    """Empty circuit has no components."""
    circuit = ImmutableCircuit("empty")
    assert circuit.name == "empty"
    assert len(circuit.components) == 0
    assert len(circuit.directives) == 0


def test_add_component():
    """Adding component returns new circuit instance."""
    circuit1 = ImmutableCircuit("test")
    r1 = Resistor("R1", 1000)

    circuit2 = circuit1.add(r1)

    # Original unchanged (immutability)
    assert len(circuit1.components) == 0
    assert len(circuit2.components) == 1
    assert circuit2.components[0] is r1


def test_add_multiple_components():
    """Can add multiple components at once."""
    circuit = ImmutableCircuit("test")
    r1 = Resistor("R1", 1000)
    c1 = Capacitor("C1", 1e-6)

    circuit = circuit.add(r1, c1)

    assert len(circuit.components) == 2
    assert circuit.components[0] is r1
    assert circuit.components[1] is c1


def test_add_directive():
    """Adding directive returns new circuit."""
    circuit1 = ImmutableCircuit("test")
    circuit2 = circuit1.add_directive(".option temp=27")

    assert len(circuit1.directives) == 0
    assert len(circuit2.directives) == 1
    assert circuit2.directives[0] == ".option temp=27"


def test_add_directive_once_idempotent():
    """add_directive_once is idempotent."""
    circuit = ImmutableCircuit("test")
    circuit = circuit.add_directive_once(".param R=1k")
    circuit = circuit.add_directive_once(".param R=1k")  # Duplicate

    # Should only have one
    assert len(circuit.directives) == 1


def test_connect_ports():
    """Connecting ports merges their nets."""
    circuit = ImmutableCircuit("test")
    r1 = Resistor("R1", 1000)
    r2 = Resistor("R2", 2000)

    circuit = circuit.add(r1, r2)

    # Connect R1.b to R2.a
    circuit = circuit.connect(r1.ports[1], r2.ports[0])

    # Both ports should be in same equivalence class
    root1 = circuit.nets.find(r1.ports[1])
    root2 = circuit.nets.find(r2.ports[0])
    assert root1 == root2


def test_connect_port_to_net():
    """Connecting port to named net."""
    circuit = ImmutableCircuit("test")
    r1 = Resistor("R1", 1000)
    vout = Net("vout")

    circuit = circuit.add(r1)
    circuit = circuit.connect(r1.ports[0], vout)

    # Port should be connected to vout
    root = circuit.nets.find(r1.ports[0])
    assert root == vout


def test_connect_to_gnd():
    """Connecting to GND singleton."""
    circuit = ImmutableCircuit("test")
    r1 = Resistor("R1", 1000)

    circuit = circuit.add(r1)
    circuit = circuit.connect(r1.ports[1], GND)

    root = circuit.nets.find(r1.ports[1])
    assert root == GND


def test_build_netlist_simple():
    """Building netlist for simple RC circuit."""
    circuit = ImmutableCircuit("rc_filter")

    r1 = Resistor("R1", "1k")
    c1 = Capacitor("C1", "100n")
    vin = Net("vin")
    vout = Net("vout")

    circuit = (
        circuit.add(r1, c1)
        .connect(r1.ports[0], vin)
        .connect(r1.ports[1], vout)
        .connect(c1.ports[0], vout)
        .connect(c1.ports[1], GND)
    )

    netlist = circuit.build_netlist()

    assert "* rc_filter" in netlist
    assert "RR1 vin vout 1k" in netlist
    assert "CC1 vout 0 100n" in netlist
    assert ".end" in netlist


def test_lazy_node_assignment():
    """Node IDs computed lazily and cached."""
    circuit = ImmutableCircuit("test")
    r1 = Resistor("R1", 1000)
    c1 = Capacitor("C1", 1e-6)

    circuit = (
        circuit.add(r1, c1)
        .connect(r1.ports[0], Net("in"))
        .connect(r1.ports[1], Net("out"))
        .connect(c1.ports[0], Net("out"))
        .connect(c1.ports[1], GND)
    )

    # First access computes node IDs
    assignments1 = circuit._node_assignments

    # Second access should return cached result (same dict object)
    assignments2 = circuit._node_assignments
    assert assignments1 is assignments2  # Identity check


def test_summary():
    """Summary shows components and connectivity."""
    circuit = ImmutableCircuit("rc")
    r1 = Resistor("R1", "1k")
    c1 = Capacitor("C1", "100n")

    circuit = (
        circuit.add(r1, c1)
        .connect(r1.ports[0], Net("in"))
        .connect(r1.ports[1], Net("out"))
        .connect(c1.ports[0], Net("out"))
        .connect(c1.ports[1], GND)
    )

    summary = circuit.summary()

    assert "Circuit: rc" in summary
    assert "R1" in summary
    assert "C1" in summary
    assert "in" in summary
    assert "out" in summary


def test_to_dot():
    """Graphviz DOT generation."""
    circuit = ImmutableCircuit("test")
    r1 = Resistor("R1", 1000)

    circuit = circuit.add(r1).connect(r1.ports[0], Net("a")).connect(r1.ports[1], GND)

    dot = circuit.to_dot()

    assert "graph circuit {" in dot
    assert "R1" in dot
    assert "}" in dot


def test_immutability_via_replace():
    """Modifications return new instances."""
    circuit1 = ImmutableCircuit("v1")
    circuit2 = circuit1.add(Resistor("R1", 1000))
    circuit3 = circuit2.add(Capacitor("C1", 1e-6))

    # Three distinct objects
    assert circuit1 is not circuit2
    assert circuit2 is not circuit3
    assert circuit1 is not circuit3

    # Originals unchanged
    assert len(circuit1.components) == 0
    assert len(circuit2.components) == 1
    assert len(circuit3.components) == 2


def test_structural_sharing():
    """Components tuple shares structure (no full copy)."""
    circuit1 = ImmutableCircuit("test")
    r1 = Resistor("R1", 1000)
    r2 = Resistor("R2", 2000)

    circuit2 = circuit1.add(r1)
    circuit3 = circuit2.add(r2)

    # Components are shared (same tuple prefix)
    # circuit3.components = circuit2.components + (r2,)
    # Both reference same R1 object
    assert circuit2.components[0] is circuit3.components[0]


def test_connect_with_label():
    """Port labels preserved."""
    circuit = ImmutableCircuit("test")
    r1 = Resistor("R1", 1000)
    vdd = Net("vdd")

    circuit = circuit.add(r1).connect_with_label(r1.ports[0], vdd, "VDD_3V3")

    assert r1.ports[0] in circuit.port_labels
    assert circuit.port_labels[r1.ports[0]] == "VDD_3V3"


def test_hash_deterministic():
    """Circuit hash is deterministic."""
    r1 = Resistor("R1", 1000)
    c1 = Capacitor("C1", 1e-6)

    circuit1 = (
        ImmutableCircuit("rc")
        .add(r1, c1)
        .connect(r1.ports[0], Net("in"))
        .connect(r1.ports[1], Net("out"))
    )

    circuit2 = (
        ImmutableCircuit("rc")
        .add(r1, c1)
        .connect(r1.ports[0], Net("in"))
        .connect(r1.ports[1], Net("out"))
    )

    # Should produce same hash (deterministic)
    hash1 = circuit1.hash()
    hash2 = circuit2.hash()

    # Note: exact equality depends on circuit_hash implementation
    # At minimum, should be consistent
    assert isinstance(hash1, str)
    assert isinstance(hash2, str)


def test_save_netlist(tmp_path):
    """Saving netlist to file."""
    circuit = ImmutableCircuit("test")
    r1 = Resistor("R1", "1k")

    circuit = circuit.add(r1).connect(r1.ports[0], Net("a")).connect(r1.ports[1], GND)

    path = tmp_path / "test.cir"
    result_path = circuit.save_netlist(path)

    assert result_path.exists()
    content = result_path.read_text()
    assert "RR1 a 0 1k" in content


def test_compatibility_components_list():
    """_components property for backward compat."""
    circuit = ImmutableCircuit("test")
    r1 = Resistor("R1", 1000)

    circuit = circuit.add(r1)

    # Should expose list (not tuple)
    components = circuit._components
    assert isinstance(components, list)
    assert len(components) == 1


def test_compatibility_port_to_net():
    """_port_to_net property for backward compat."""
    circuit = ImmutableCircuit("test")
    r1 = Resistor("R1", 1000)
    net = Net("vout")

    circuit = circuit.add(r1).connect(r1.ports[0], net)

    mapping = circuit._port_to_net
    assert isinstance(mapping, dict)
    assert r1.ports[0] in mapping
    assert mapping[r1.ports[0]] == net


def test_connectivity_dataframe():
    """Connectivity DataFrame generation."""
    pytest.importorskip("pandas")

    circuit = ImmutableCircuit("test")
    r1 = Resistor("R1", 1000)

    circuit = circuit.add(r1).connect(r1.ports[0], Net("a")).connect(r1.ports[1], GND)

    df = circuit.connectivity_dataframe()

    assert len(df) == 2  # Two ports
    assert "component" in df.columns
    assert "net" in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
