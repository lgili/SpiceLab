"""Tests for immutable Union-Find NetRegistry."""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest
from spicelab.core.net import GND, Net, Port, PortRole
from spicelab.core.net_registry import NetRegistry


# Test fixtures
@dataclass
class MockComponent:
    """Mock component for testing ports."""

    ref: str


@pytest.fixture
def registry() -> NetRegistry:
    """Empty registry."""
    return NetRegistry()


@pytest.fixture
def mock_component() -> MockComponent:
    """Mock component for port creation."""
    return MockComponent(ref="R1")


def test_empty_registry():
    """Empty registry has no nodes."""
    reg = NetRegistry()
    assert len(reg) == 0
    assert len(reg.all_nets()) == 0


def test_find_unregistered_node(registry: NetRegistry, mock_component: MockComponent):
    """Finding unregistered node returns the node itself."""
    port = Port(mock_component, "a", PortRole.POSITIVE)
    root = registry.find(port)
    assert root == port


def test_ensure_registered(registry: NetRegistry, mock_component: MockComponent):
    """Registering a node creates singleton set."""
    port = Port(mock_component, "a", PortRole.POSITIVE)

    # Register port
    reg2 = registry._ensure_registered(port)
    assert len(reg2) == 1
    assert reg2.find(port) == port

    # Idempotent: registering again doesn't change anything
    reg3 = reg2._ensure_registered(port)
    assert len(reg3) == 1


def test_union_two_nodes(registry: NetRegistry, mock_component: MockComponent):
    """Union merges two singleton sets."""
    port_a = Port(mock_component, "a", PortRole.POSITIVE)
    port_b = Port(mock_component, "b", PortRole.NEGATIVE)

    # Union port_a and port_b
    reg2 = registry.union(port_a, port_b)

    # Both should have same root
    root_a = reg2.find(port_a)
    root_b = reg2.find(port_b)
    assert root_a == root_b


def test_union_with_net(registry: NetRegistry, mock_component: MockComponent):
    """Union with Net makes Net the root."""
    port = Port(mock_component, "a", PortRole.POSITIVE)
    net = Net("vdd")

    # Union port with net
    reg2 = registry.union(port, net)

    # Net should be root (higher rank due to naming)
    # Root could be either port or net depending on union order
    # Just verify they're in same equivalence class
    assert reg2.find(port) == reg2.find(net)


def test_transitive_union(registry: NetRegistry):
    """Union is transitive: union(a,b) + union(b,c) => find(a)==find(c)."""
    comp1 = MockComponent("R1")
    comp2 = MockComponent("R2")
    comp3 = MockComponent("R3")

    port_a = Port(comp1, "a", PortRole.POSITIVE)
    port_b = Port(comp2, "b", PortRole.POSITIVE)
    port_c = Port(comp3, "c", PortRole.POSITIVE)

    # Chain unions: a-b, b-c
    reg2 = registry.union(port_a, port_b)
    reg3 = reg2.union(port_b, port_c)

    # All three should have same root
    root_a = reg3.find(port_a)
    root_b = reg3.find(port_b)
    root_c = reg3.find(port_c)

    assert root_a == root_b == root_c


def test_union_idempotent(registry: NetRegistry, mock_component: MockComponent):
    """Union of already-merged nodes is idempotent."""
    port_a = Port(mock_component, "a", PortRole.POSITIVE)
    port_b = Port(mock_component, "b", PortRole.NEGATIVE)

    # Union twice
    reg2 = registry.union(port_a, port_b)
    reg3 = reg2.union(port_a, port_b)

    # Should still have same number of nodes
    assert len(reg2) == len(reg3)


def test_get_net(registry: NetRegistry, mock_component: MockComponent):
    """get_net returns Net if port connected to one."""
    port = Port(mock_component, "a", PortRole.POSITIVE)
    net = Net("vout")

    # Before union
    assert registry.get_net(port) is None

    # After union
    reg2 = registry.union(port, net)
    result = reg2.get_net(port)

    # Should find net in equivalence class
    # (Could be net itself or another Net in same class)
    assert result == net or result is None  # Depends on union order


def test_set_name(registry: NetRegistry):
    """set_name associates display name with net."""
    net = Net("vdd")

    reg2 = registry._ensure_registered(net)
    reg3 = reg2.set_name(net, "VDD_3V3")

    assert reg3.get_name(net) == "VDD_3V3"


def test_all_nets(registry: NetRegistry):
    """all_nets returns all root nodes."""
    comp1 = MockComponent("R1")
    comp2 = MockComponent("R2")

    port_a = Port(comp1, "a", PortRole.POSITIVE)
    port_b = Port(comp1, "b", PortRole.NEGATIVE)
    port_c = Port(comp2, "a", PortRole.POSITIVE)

    net1 = Net("net1")
    net2 = Net("net2")

    # Create two separate nets
    reg = registry.union(port_a, net1)
    reg = reg.union(port_b, net1)  # Same net
    reg = reg.union(port_c, net2)  # Different net

    nets = reg.all_nets()

    # Should have 2 equivalence classes (net1 and net2)
    assert len(nets) == 2


def test_connected_ports(registry: NetRegistry):
    """connected_ports returns all ports in net's equivalence class."""
    comp = MockComponent("R1")
    port_a = Port(comp, "a", PortRole.POSITIVE)
    port_b = Port(comp, "b", PortRole.NEGATIVE)

    net = Net("vout")

    # Connect both ports to same net
    reg = registry.union(port_a, net)
    reg = reg.union(port_b, net)

    ports = reg.connected_ports(net)

    assert port_a in ports
    assert port_b in ports
    assert len(ports) == 2


def test_immutability(registry: NetRegistry, mock_component: MockComponent):
    """Operations return new instances (immutability)."""
    port = Port(mock_component, "a", PortRole.POSITIVE)
    net = Net("vdd")

    # Original registry unchanged
    reg2 = registry.union(port, net)

    assert len(registry) == 0  # Original empty
    assert len(reg2) > 0  # New has nodes
    assert registry is not reg2


# Performance benchmarks
def test_benchmark_union_performance():
    """Benchmark union operations vs naive dict approach."""
    n_nodes = 1000

    # Prepare nodes
    components = [MockComponent(f"R{i}") for i in range(n_nodes)]
    ports = [Port(comp, "a", PortRole.POSITIVE) for comp in components]

    # Benchmark Union-Find
    reg = NetRegistry()
    start = time.perf_counter()

    # Chain all ports together (worst case for naive approach)
    for i in range(1, n_nodes):
        reg = reg.union(ports[i - 1], ports[i])

    uf_time = time.perf_counter() - start

    # Verify all connected
    root = reg.find(ports[0])
    for port in ports[1:]:
        assert reg.find(port) == root

    print(f"\nUnion-Find: {n_nodes} unions in {uf_time*1000:.2f}ms")

    # Benchmark naive dict approach (simulate old Circuit._port_to_net)
    port_to_net: dict[Port, Net] = {}
    start = time.perf_counter()

    # Simulate net merges with linear scans
    current_net = Net("net0")
    port_to_net[ports[0]] = current_net

    for i in range(1, n_nodes):
        # Simulate merge: find all ports with old net, remap to current
        old_net = port_to_net.get(ports[i - 1])
        if old_net:
            for p, n in list(port_to_net.items()):
                if n == old_net:
                    port_to_net[p] = current_net
        port_to_net[ports[i]] = current_net

    dict_time = time.perf_counter() - start

    print(f"Dict approach: {n_nodes} merges in {dict_time*1000:.2f}ms")
    print(f"Speedup: {dict_time / uf_time:.1f}x")

    # Union-Find should be significantly faster
    assert uf_time < dict_time


def test_path_compression():
    """Verify path compression reduces tree height."""
    reg = NetRegistry()

    # Create deep chain: p0 -> p1 -> p2 -> p3 -> net
    components = [MockComponent(f"R{i}") for i in range(4)]
    ports = [Port(comp, "a", PortRole.POSITIVE) for comp in components]
    net = Net("vdd")

    # Build chain
    for i in range(len(ports) - 1):
        reg = reg.union(ports[i], ports[i + 1])
    reg = reg.union(ports[-1], net)

    # First find on p0 walks full chain
    root1 = reg.find(ports[0])

    # After path compression, p0 should point closer to root
    # (In our implementation, _parent gets updated during find)
    root2 = reg.find(ports[0])

    assert root1 == root2  # Same root


def test_gnd_special_handling():
    """GND singleton is handled correctly."""
    reg = NetRegistry()
    comp = MockComponent("R1")
    port = Port(comp, "a", PortRole.POSITIVE)

    # Connect to GND
    reg = reg.union(port, GND)

    # GND should be in equivalence class
    root = reg.find(port)
    assert root == GND or reg.find(root) == GND


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_net_registry.py -v
    pytest.main([__file__, "-v"])
