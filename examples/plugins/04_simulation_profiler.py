"""Example: Simulation Profiler Plugin

This example demonstrates how to use the SimulationProfilerPlugin to
profile simulation performance and identify bottlenecks.

The plugin tracks:
- Simulation time
- Netlist build time
- Analysis time
- Cache hit/miss rates
- Memory usage (if psutil is available)
"""

import time

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.plugins.examples import SimulationProfilerPlugin
from spicelab.plugins.hooks import HookManager, HookType


def create_test_circuit(name: str = "test_circuit", n_stages: int = 1) -> Circuit:
    """Create a test circuit with configurable complexity."""
    circuit = Circuit(name)

    vin = Vdc("Vin", 5.0)
    circuit.add(vin)
    circuit.connect(vin.ports[0], Net("in"))
    circuit.connect(vin.ports[1], GND)

    prev_net = "in"
    for i in range(n_stages):
        r = Resistor(f"R{i+1}", "1k")
        c = Capacitor(f"C{i+1}", "1u")
        circuit.add(r, c)

        next_net = f"n{i+1}" if i < n_stages - 1 else "out"
        circuit.connect(r.ports[0], Net(prev_net))
        circuit.connect(r.ports[1], Net(next_net))
        circuit.connect(c.ports[0], Net(next_net))
        circuit.connect(c.ports[1], GND)
        prev_net = next_net

    return circuit


def example_basic_profiling():
    """Basic profiling demonstration."""
    print("=" * 60)
    print("Example 1: Basic Profiling")
    print("=" * 60)

    plugin = SimulationProfilerPlugin()
    plugin.activate()

    # Get hook manager
    hook_manager = HookManager.get_instance()

    # Simulate some operations
    circuit = create_test_circuit("profiled_circuit")

    print("\nSimulating workflow with profiling hooks...")

    # Simulate netlist build
    hook_manager.trigger(HookType.PRE_NETLIST_BUILD, circuit=circuit)
    time.sleep(0.05)  # Simulate work
    _ = circuit.build_netlist()
    hook_manager.trigger(HookType.POST_NETLIST_BUILD, circuit=circuit, netlist="...")

    # Simulate validation
    hook_manager.trigger(HookType.PRE_VALIDATION, circuit=circuit)
    time.sleep(0.02)  # Simulate work
    hook_manager.trigger(HookType.POST_VALIDATION, circuit=circuit, errors=[], warnings=[])

    # Simulate simulation
    hook_manager.trigger(HookType.PRE_SIMULATION, circuit=circuit, analyses=[])
    time.sleep(0.1)  # Simulate work
    hook_manager.trigger(HookType.POST_SIMULATION, circuit=circuit, result=None)

    # Get and print report
    print("\n" + plugin.get_report())

    plugin.deactivate()


def example_stats_analysis():
    """Analyze profiling statistics."""
    print("\n" + "=" * 60)
    print("Example 2: Statistics Analysis")
    print("=" * 60)

    plugin = SimulationProfilerPlugin()
    plugin.activate()

    hook_manager = HookManager.get_instance()

    # Run multiple simulations to gather statistics
    print("\nRunning 5 simulated operations...")
    for i in range(5):
        circuit = create_test_circuit(f"circuit_{i}", n_stages=i + 1)

        hook_manager.trigger(HookType.PRE_SIMULATION, circuit=circuit, analyses=[])
        time.sleep(0.02 * (i + 1))  # Varying simulation times
        hook_manager.trigger(HookType.POST_SIMULATION, circuit=circuit, result=None)

    # Get statistics
    stats = plugin.get_stats()
    print("\nAggregated Statistics:")
    for operation, stat in stats.items():
        print(f"\n  {operation}:")
        print(f"    Count: {stat.count}")
        print(f"    Total: {stat.total_ms:.2f} ms")
        print(f"    Average: {stat.avg_ms:.2f} ms")
        print(f"    Min: {stat.min_ms:.2f} ms")
        print(f"    Max: {stat.max_ms:.2f} ms")

    plugin.deactivate()


def example_cache_tracking():
    """Track cache hit/miss rates."""
    print("\n" + "=" * 60)
    print("Example 3: Cache Tracking")
    print("=" * 60)

    plugin = SimulationProfilerPlugin()
    plugin.activate()

    hook_manager = HookManager.get_instance()

    # Simulate cache operations
    print("\nSimulating cache operations...")

    # First run - cache misses
    for i in range(3):
        hook_manager.trigger(HookType.CACHE_MISS, key=f"sim_key_{i}")

    # Second run - cache hits
    for i in range(7):
        hook_manager.trigger(HookType.CACHE_HIT, key=f"sim_key_{i % 3}")

    # Get cache stats
    cache_stats = plugin.get_cache_stats()
    print("\nCache Statistics:")
    print(f"  Hits: {cache_stats['hits']}")
    print(f"  Misses: {cache_stats['misses']}")
    print(f"  Total: {cache_stats['total']}")
    print(f"  Hit Rate: {cache_stats['hit_rate']:.1%}")

    plugin.deactivate()


def example_slowest_operations():
    """Find the slowest operations."""
    print("\n" + "=" * 60)
    print("Example 4: Finding Slowest Operations")
    print("=" * 60)

    plugin = SimulationProfilerPlugin()
    plugin.activate()

    hook_manager = HookManager.get_instance()

    # Run operations with varying times
    print("\nRunning operations with varying durations...")

    delays = [0.01, 0.05, 0.02, 0.15, 0.03, 0.08, 0.01, 0.12, 0.04, 0.06]
    for i, delay in enumerate(delays):
        circuit = create_test_circuit(f"circuit_{i}")
        hook_manager.trigger(HookType.PRE_SIMULATION, circuit=circuit, analyses=[])
        time.sleep(delay)
        hook_manager.trigger(HookType.POST_SIMULATION, circuit=circuit, result=None)

    # Get slowest operations
    slowest = plugin.get_slowest(n=5)
    print("\nTop 5 Slowest Operations:")
    for i, record in enumerate(slowest, 1):
        print(f"  {i}. {record.operation}: {record.duration_ms:.2f} ms")

    # Get slowest of specific type
    slowest_sims = plugin.get_slowest(operation="simulation", n=3)
    print("\nTop 3 Slowest Simulations:")
    for i, record in enumerate(slowest_sims, 1):
        print(f"  {i}. {record.duration_ms:.2f} ms")

    plugin.deactivate()


def example_export_profile():
    """Export profiling data to file."""
    print("\n" + "=" * 60)
    print("Example 5: Export Profile Data")
    print("=" * 60)

    plugin = SimulationProfilerPlugin()
    plugin.activate()

    hook_manager = HookManager.get_instance()

    # Generate some data
    for i in range(3):
        circuit = create_test_circuit(f"circuit_{i}")
        hook_manager.trigger(HookType.PRE_SIMULATION, circuit=circuit, analyses=[])
        time.sleep(0.03)
        hook_manager.trigger(HookType.POST_SIMULATION, circuit=circuit, result=None)

    # Export to JSON
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "profile_data.json"
        plugin.export_profile(export_path)

        print(f"\nExported profile to: {export_path}")
        print("\nExported JSON content:")
        import json

        data = json.loads(export_path.read_text())
        print(json.dumps(data, indent=2)[:500] + "...")

    plugin.deactivate()


def example_memory_usage():
    """Track memory usage (requires psutil)."""
    print("\n" + "=" * 60)
    print("Example 6: Memory Usage Tracking")
    print("=" * 60)

    plugin = SimulationProfilerPlugin()
    plugin.activate()

    mem = plugin.get_memory_usage()

    if mem:
        print("\nCurrent Memory Usage:")
        print(f"  RSS (Resident Set Size): {mem['rss_mb']:.2f} MB")
        print(f"  VMS (Virtual Memory Size): {mem['vms_mb']:.2f} MB")
        print(f"  Memory Percent: {mem['percent']:.1f}%")
    else:
        print("\nNote: Install 'psutil' for memory tracking:")
        print("  pip install psutil")

    plugin.deactivate()


def example_configuration():
    """Configure profiler options."""
    print("\n" + "=" * 60)
    print("Example 7: Configuration Options")
    print("=" * 60)

    plugin = SimulationProfilerPlugin()

    # Show configuration options
    print("\nConfiguration Options:")
    print("""
    plugin.configure({
        "track_memory": True,       # Track memory usage
        "max_records": 1000,        # Maximum timing records to keep
        "auto_report": False,       # Print report on deactivate
        "report_threshold_ms": 100, # Alert on slow operations
    })
    """)

    # Configure for auto-reporting
    plugin.configure({
        "auto_report": True,
        "report_threshold_ms": 50,
    })

    print("Configured with auto_report=True and 50ms threshold")
    print("Slow operations (>50ms) will be automatically logged.")


def example_reset_profiler():
    """Reset profiler data."""
    print("\n" + "=" * 60)
    print("Example 8: Reset Profiler")
    print("=" * 60)

    plugin = SimulationProfilerPlugin()
    plugin.activate()

    hook_manager = HookManager.get_instance()

    # Generate some data
    for i in range(5):
        hook_manager.trigger(HookType.CACHE_HIT, key=f"key_{i}")

    print(f"Before reset: {plugin.get_cache_stats()['hits']} cache hits")

    # Reset
    plugin.reset()

    print(f"After reset: {plugin.get_cache_stats()['hits']} cache hits")
    print("\nReset clears all timing records and statistics.")

    plugin.deactivate()


def example_recent_records():
    """View recent timing records."""
    print("\n" + "=" * 60)
    print("Example 9: Recent Records")
    print("=" * 60)

    plugin = SimulationProfilerPlugin()
    plugin.activate()

    hook_manager = HookManager.get_instance()

    # Generate some operations
    operations = ["netlist_build", "validation", "simulation"]
    for op in operations:
        circuit = create_test_circuit("test")
        if op == "netlist_build":
            hook_manager.trigger(HookType.PRE_NETLIST_BUILD, circuit=circuit)
            time.sleep(0.01)
            hook_manager.trigger(HookType.POST_NETLIST_BUILD, circuit=circuit, netlist="")
        elif op == "validation":
            hook_manager.trigger(HookType.PRE_VALIDATION, circuit=circuit)
            time.sleep(0.015)
            hook_manager.trigger(HookType.POST_VALIDATION, circuit=circuit, errors=[], warnings=[])
        else:
            hook_manager.trigger(HookType.PRE_SIMULATION, circuit=circuit, analyses=[])
            time.sleep(0.02)
            hook_manager.trigger(HookType.POST_SIMULATION, circuit=circuit, result=None)

    # Get recent records
    recent = plugin.get_recent_records(n=5)
    print("\nMost Recent Operations:")
    for record in recent:
        print(f"  {record.operation}: {record.duration_ms:.2f} ms")

    plugin.deactivate()


if __name__ == "__main__":
    example_basic_profiling()
    example_stats_analysis()
    example_cache_tracking()
    example_slowest_operations()
    example_export_profile()
    example_memory_usage()
    example_configuration()
    example_reset_profiler()
    example_recent_records()

    print("\n" + "=" * 60)
    print("Simulation Profiler Plugin Examples Complete!")
    print("=" * 60)
