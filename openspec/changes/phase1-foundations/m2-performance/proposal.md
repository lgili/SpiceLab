# M2: Performance Baseline

**Status:** Proposed
**Priority:** 🟠 HIGH
**Estimated Duration:** 4-6 weeks
**Dependencies:** M1 (stress test infrastructure)

## Problem Statement

SpiceLab currently lacks performance benchmarks and optimization for large-scale circuits. Without profiling data and optimization, the library is not viable for industrial circuits with thousands of components, limiting adoption in professional settings.

### Current Gaps
- ❌ No profiling (bottlenecks unknown)
- ❌ No comparative benchmarks (vs PySpice/spicelib)
- ❌ Netlist building O(n²) for large circuits
- ❌ No lazy evaluation (all components instantiated eagerly)
- ❌ Single-threaded RAW file parsing
- ❌ No memory leak detection in long-running jobs
- ❌ No performance tracking in CI

### Impact
- **Industrial Use:** Impractical for circuits >1k components (slow netlist generation)
- **Monte Carlo:** Memory leaks in long-running simulations
- **User Experience:** Users abandon tool due to poor performance
- **Maintenance:** No visibility into performance regressions

## Objectives

1. **Profile** entire codebase with cProfile and py-spy to identify bottlenecks
2. **Benchmark** against PySpice and spicelib with realistic circuits
3. **Optimize netlist building** from O(n²) to O(n) using graph-based approach
4. **Implement lazy loading** for component instantiation
5. **Parallelize RAW file parsing** for multi-signal files
6. **Detect and fix memory leaks** in Monte Carlo simulations
7. **Track performance in CI** to prevent regressions
8. **Target:** 10x faster netlist build, 50% less memory in Monte Carlo

## Technical Design

### 1. Comprehensive Profiling

**Strategy:** Profile all critical paths to identify bottlenecks.

#### Profiling Approach
```python
# tools/profile_netlist_build.py
import cProfile
import pstats
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Capacitor

def build_large_circuit(n: int) -> Circuit:
    """Build circuit with n components for profiling."""
    circuit = Circuit(f"profile_{n}")

    # Create RC ladder network
    for i in range(n // 2):
        r = Resistor(f"R{i}", "1k")
        c = Capacitor(f"C{i}", "100n")
        circuit.add(r)
        circuit.add(c)
        # Connect in ladder topology
        if i > 0:
            circuit.connect(r.ports[0], f"n{i-1}")
        circuit.connect(r.ports[1], f"n{i}")
        circuit.connect(c.ports[0], f"n{i}")
        circuit.connect(c.ports[1], "0")

    return circuit

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    # Profile netlist generation for 10k components
    circuit = build_large_circuit(10000)
    netlist = circuit.build_netlist()

    profiler.disable()

    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(50)  # Top 50 functions

    # Save for visualization
    stats.dump_stats('profile_netlist_10k.prof')
```

#### py-spy Continuous Profiling
```bash
# Profile running simulation
py-spy record -o profile.svg -- python examples/monte_carlo_large.py

# Profile specific function
py-spy top -- python -m pytest tests/stress/test_large_circuits.py::test_10k_components
```

### 2. Comparative Benchmarks

**Strategy:** Measure SpiceLab vs competitors on identical circuits.

#### Benchmark Suite
```python
# benchmarks/bench_netlist_generation.py
import time
import pytest
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor

# PySpice comparison (optional)
try:
    from PySpice.Spice.Netlist import Circuit as PySpiceCircuit
    PYSPICE_AVAILABLE = True
except ImportError:
    PYSPICE_AVAILABLE = False

@pytest.mark.benchmark
@pytest.mark.parametrize("n_components", [10, 100, 1000, 5000, 10000])
def test_spicelab_netlist_generation(benchmark, n_components):
    """Benchmark SpiceLab netlist generation."""
    def setup():
        circuit = Circuit(f"bench_{n_components}")
        for i in range(n_components):
            r = Resistor(f"R{i}", "1k")
            circuit.add(r)
        return circuit

    def run(circuit):
        return circuit.build_netlist()

    circuit = setup()
    result = benchmark(run, circuit)
    assert len(result) > 0

@pytest.mark.benchmark
@pytest.mark.skipif(not PYSPICE_AVAILABLE, reason="PySpice not installed")
@pytest.mark.parametrize("n_components", [10, 100, 1000, 5000, 10000])
def test_pyspice_netlist_generation(benchmark, n_components):
    """Benchmark PySpice netlist generation for comparison."""
    def setup():
        circuit = PySpiceCircuit("bench")
        for i in range(n_components):
            circuit.R(i, i, i+1, "1k")
        return circuit

    def run(circuit):
        return str(circuit)

    circuit = setup()
    result = benchmark(run, circuit)
    assert len(result) > 0

# benchmarks/bench_raw_parsing.py
@pytest.mark.benchmark
def test_raw_file_parsing(benchmark, tmp_path):
    """Benchmark RAW file parsing."""
    # Create large RAW file (1000 signals, 10000 points)
    raw_file = _create_test_raw(tmp_path, n_signals=1000, n_points=10000)

    from spicelab.io.raw import read_raw

    result = benchmark(read_raw, raw_file)
    assert result is not None
```

#### Benchmark CI Integration
```yaml
# .github/workflows/benchmark.yml
name: Performance Benchmarks

on:
  push:
    branches: [main]
  pull_request:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v1

      - name: Run benchmarks
        run: |
          uv run pytest benchmarks/ --benchmark-json=output.json

      - name: Store benchmark result
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'pytest'
          output-file-path: output.json
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true
          alert-threshold: '150%'
          comment-on-alert: true
          fail-on-alert: false
```

### 3. Graph-Based Netlist Optimization

**Problem:** Current implementation rebuilds net connectivity on every query (O(n²)).

**Solution:** Build dependency graph once, traverse efficiently.

#### Implementation
```python
# spicelab/core/circuit.py (OPTIMIZED)
from __future__ import annotations
import networkx as nx
from typing import Dict, Set

class Circuit:
    """Circuit with optimized graph-based connectivity."""

    def __init__(self, name: str):
        self.name = name
        self._components: Dict[str, Component] = {}
        self._nets: Dict[str, Net] = {}
        # NEW: Connectivity graph (cached)
        self._graph: nx.Graph | None = None
        self._graph_dirty: bool = True

    def add(self, component: Component) -> None:
        """Add component and mark graph dirty."""
        self._components[component.ref] = component
        self._graph_dirty = True

    def connect(self, port: Port, net: Net | str) -> None:
        """Connect port to net and mark graph dirty."""
        net_obj = self._get_or_create_net(net)
        port._net = net_obj
        net_obj._ports.add(port)
        self._graph_dirty = True

    @property
    def graph(self) -> nx.Graph:
        """Get connectivity graph (cached)."""
        if self._graph_dirty or self._graph is None:
            self._rebuild_graph()
        return self._graph

    def _rebuild_graph(self) -> None:
        """Rebuild connectivity graph from current state."""
        G = nx.Graph()

        # Add nodes (nets)
        for net_name in self._nets:
            G.add_node(net_name)

        # Add edges (components connect nets)
        for comp in self._components.values():
            nets = [port.net.name for port in comp.ports if port.net]
            for i in range(len(nets) - 1):
                G.add_edge(nets[i], nets[i+1], component=comp.ref)

        self._graph = G
        self._graph_dirty = False

    def build_netlist(self) -> str:
        """Build netlist using graph traversal (O(n) instead of O(n²))."""
        lines = [f"* {self.name}"]

        # Sort components by graph order (depth-first)
        sorted_components = self._topological_sort_components()

        for comp in sorted_components:
            lines.append(comp.spice_card())

        return "\n".join(lines)

    def _topological_sort_components(self) -> list[Component]:
        """Sort components in netlist-friendly order using graph."""
        G = self.graph

        # Components attached to each net
        net_components: Dict[str, Set[str]] = {}
        for comp in self._components.values():
            for port in comp.ports:
                if port.net:
                    net_components.setdefault(port.net.name, set()).add(comp.ref)

        # Start from GND, traverse breadth-first
        visited_nets = set()
        visited_comps = set()
        ordered_comps = []
        queue = ["0"]  # Start from GND

        while queue:
            net = queue.pop(0)
            if net in visited_nets:
                continue
            visited_nets.add(net)

            # Add all components connected to this net
            for comp_ref in net_components.get(net, []):
                if comp_ref not in visited_comps:
                    visited_comps.add(comp_ref)
                    comp = self._components[comp_ref]
                    ordered_comps.append(comp)

                    # Queue neighboring nets
                    for port in comp.ports:
                        if port.net and port.net.name not in visited_nets:
                            queue.append(port.net.name)

        return ordered_comps
```

**Performance Gain:** O(n) vs O(n²) for large circuits.

### 4. Lazy Component Instantiation

**Problem:** All component attributes are computed eagerly, even if never used.

**Solution:** Use properties with caching for expensive operations.

#### Implementation
```python
# spicelab/core/components.py (OPTIMIZED)
from functools import cached_property

class Component(Protocol):
    """Base component with lazy evaluation."""

    ref: str
    _spice_card_cache: str | None = None

    @cached_property
    def equivalent_impedance(self) -> complex:
        """Compute impedance (expensive, computed once)."""
        # Complex calculation...
        return self._compute_impedance()

    def spice_card(self) -> str:
        """Generate SPICE card (cached)."""
        if self._spice_card_cache is None:
            self._spice_card_cache = self._build_spice_card()
        return self._spice_card_cache

    def invalidate_cache(self) -> None:
        """Clear cached values after modification."""
        self._spice_card_cache = None
        if hasattr(self, '__dict__'):
            # Clear cached_property values
            for key in list(self.__dict__.keys()):
                if key.startswith('_cached_'):
                    delattr(self, key)

class Resistor:
    """Resistor with lazy evaluation."""

    def __init__(self, ref: str, resistance: float | str):
        self.ref = ref
        self._resistance_raw = resistance
        self._resistance_value: float | None = None

    @property
    def resistance(self) -> float:
        """Parse resistance value lazily."""
        if self._resistance_value is None:
            from spicelab.core.units import parse_value
            self._resistance_value = parse_value(self._resistance_raw)
        return self._resistance_value
```

**Memory Gain:** ~30% reduction for large circuits with many unused attributes.

### 5. Parallel RAW File Parsing

**Problem:** RAW files with 1000+ signals are parsed sequentially.

**Solution:** Parse signal chunks in parallel using ProcessPoolExecutor.

#### Implementation
```python
# spicelab/io/raw.py (OPTIMIZED)
from concurrent.futures import ProcessPoolExecutor
from typing import List
import numpy as np

def read_raw_parallel(path: str, max_workers: int = 4) -> xr.Dataset:
    """Read RAW file with parallel signal parsing."""
    # Read header (fast, single-threaded)
    header = _parse_raw_header(path)
    n_signals = len(header.signals)
    n_points = header.n_points

    # Split signals into chunks
    chunk_size = max(1, n_signals // max_workers)
    chunks = [
        (i, min(i + chunk_size, n_signals))
        for i in range(0, n_signals, chunk_size)
    ]

    # Parse chunks in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_parse_signal_chunk, path, start, end, header)
            for start, end in chunks
        ]

        # Collect results
        signal_data = {}
        for future in futures:
            chunk_data = future.result()
            signal_data.update(chunk_data)

    # Assemble xarray Dataset
    return _build_dataset(signal_data, header)

def _parse_signal_chunk(
    path: str,
    start_idx: int,
    end_idx: int,
    header: RawHeader
) -> Dict[str, np.ndarray]:
    """Parse signals [start_idx:end_idx] from RAW file."""
    # Read only needed signals from binary file
    with open(path, 'rb') as f:
        f.seek(header.data_offset)

        data = {}
        for i in range(start_idx, end_idx):
            signal_name = header.signals[i]
            # Read signal binary data
            signal_bytes = f.read(header.n_points * 8)  # 8 bytes per double
            signal_values = np.frombuffer(signal_bytes, dtype=np.float64)
            data[signal_name] = signal_values

    return data
```

**Performance Gain:** 3-4x faster on machines with 4+ cores.

### 6. Memory Leak Detection

**Strategy:** Use tracemalloc and memory_profiler to find leaks.

#### Detection Tools
```python
# tools/detect_memory_leaks.py
import tracemalloc
import gc
from memory_profiler import profile

@profile
def test_monte_carlo_memory_leak():
    """Profile memory usage in Monte Carlo simulation."""
    from spicelab.orchestration.monte_carlo import monte_carlo
    from spicelab.core.circuit import Circuit
    from spicelab.core.components import Resistor
    from spicelab.analysis import AnalysisSpec
    from spicelab.distributions import NormalPct

    # Build simple circuit
    circuit = Circuit("rc")
    r1 = Resistor("R1", "10k")
    circuit.add(r1)

    # Run 1000 Monte Carlo iterations
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

    for i in range(10):  # Run in batches to detect leaks
        result = monte_carlo(
            circuit=circuit,
            mapping={r1: NormalPct(0.05)},
            n=100,
            analyses=[AnalysisSpec("op", {})],
            engine="ngspice"
        )

        # Force garbage collection
        del result
        gc.collect()

        # Check memory growth
        snapshot2 = tracemalloc.take_snapshot()
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')

        print(f"Batch {i+1} memory diff:")
        for stat in top_stats[:3]:
            print(stat)

        # Memory should be stable (not growing)
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current: {current / 1e6:.1f}MB, Peak: {peak / 1e6:.1f}MB")

    tracemalloc.stop()

if __name__ == "__main__":
    test_monte_carlo_memory_leak()
```

#### Common Leak Fixes
```python
# BEFORE (leaks file handles)
def run_simulation(circuit, analyses, engine):
    proc = subprocess.Popen([engine, netlist_path], stdout=subprocess.PIPE)
    output = proc.stdout.read()
    return parse_output(output)

# AFTER (proper cleanup)
def run_simulation(circuit, analyses, engine):
    try:
        proc = subprocess.Popen([engine, netlist_path], stdout=subprocess.PIPE)
        output = proc.stdout.read()
        return parse_output(output)
    finally:
        proc.kill()
        proc.wait()
```

### 7. Benchmark CI Tracking

**Strategy:** Track performance over time, alert on regressions.

#### GitHub Actions Benchmark Storage
```yaml
# .github/workflows/benchmark.yml
name: Continuous Benchmarking

on:
  push:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v1

      - name: Run benchmarks
        run: |
          uv run pytest benchmarks/ \
            --benchmark-only \
            --benchmark-json=benchmark.json \
            --benchmark-sort=name

      - name: Store benchmark results
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'pytest'
          output-file-path: benchmark.json
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true
          alert-threshold: '150%'
          comment-on-alert: true
          fail-on-alert: true
          alert-comment-cc-users: '@maintainer'

      - name: Upload results artifact
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmark.json
```

## Implementation Plan

### Week 1: Profiling & Baseline
- [ ] Setup cProfile and py-spy profiling tools
- [ ] Profile netlist generation (10, 100, 1k, 10k components)
- [ ] Profile RAW file parsing (various sizes)
- [ ] Profile Monte Carlo (100, 1k runs)
- [ ] Identify top 10 bottlenecks
- [ ] Document baseline performance metrics

### Week 2: Comparative Benchmarks
- [ ] Install PySpice and spicelib in test environment
- [ ] Create benchmark suite (pytest-benchmark)
- [ ] Write netlist generation benchmarks (all tools)
- [ ] Write analysis benchmarks (transient, AC, DC)
- [ ] Write Monte Carlo benchmarks
- [ ] Setup benchmark CI workflow
- [ ] Generate comparison report

### Week 3: Optimization (Netlist + Lazy)
- [ ] Implement graph-based circuit representation (NetworkX)
- [ ] Optimize netlist generation using topological sort
- [ ] Add lazy component attribute evaluation
- [ ] Implement SPICE card caching
- [ ] Add graph invalidation on modifications
- [ ] Benchmark improvements (target: 10x faster)

### Week 4: Optimization (Parallel + Memory)
- [ ] Implement parallel RAW file parsing
- [ ] Add ProcessPoolExecutor with chunking
- [ ] Optimize memory usage in large datasets
- [ ] Run memory leak detection tools
- [ ] Fix identified leaks (file handles, circular refs)
- [ ] Validate 50% memory reduction in Monte Carlo

### Week 5: Integration & Validation
- [ ] Run full benchmark suite with optimizations
- [ ] Compare before/after performance
- [ ] Validate correctness (results unchanged)
- [ ] Update documentation with performance tips
- [ ] Create performance tuning guide

### Week 6: CI & Monitoring
- [ ] Setup benchmark CI tracking
- [ ] Configure regression alerts (>150% slowdown)
- [ ] Add performance badges to README
- [ ] Create benchmark visualization dashboard
- [ ] Document performance optimization process

## Success Metrics

### Performance Targets
- [ ] Netlist generation (10k components): **<100ms** (currently ~1s, 10x improvement)
- [ ] RAW parsing (1k signals, 10k points): **<500ms** (4x improvement with parallelization)
- [ ] Monte Carlo (1k runs): **<500MB peak memory** (50% reduction from ~1GB)
- [ ] Overall speedup: **5-10x** for typical workflows

### Benchmark Coverage
- [ ] 20+ benchmark tests across all critical paths
- [ ] PySpice comparison benchmarks (10 tests)
- [ ] spicelib comparison benchmarks (optional, 5 tests)
- [ ] Memory profiling tests (5 scenarios)

### CI Integration
- [ ] Benchmark CI runs on every main push
- [ ] Regression alerts active (<150% threshold)
- [ ] Performance trends visible (GitHub Pages dashboard)
- [ ] Benchmark results archived for history

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Optimizations break correctness | CRITICAL | Extensive regression tests, validate results match original |
| Parallel parsing slower on single-core | MEDIUM | Auto-detect CPU count, fallback to serial if cores < 2 |
| Memory profiling overhead | LOW | Run profiling separately, not in CI |
| Benchmark noise/flakiness | MEDIUM | Run multiple iterations, use median, set generous thresholds |
| NetworkX dependency overhead | LOW | Make optional, fallback to naive implementation |

## Dependencies

**Required:**
- pytest-benchmark (`pip install pytest-benchmark`)
- NetworkX (`pip install networkx`)
- memory_profiler (`pip install memory-profiler`)
- py-spy (`pip install py-spy`)

**Optional (for comparison):**
- PySpice (`pip install PySpice`)
- spicelib (`pip install spicelib`)

## Future Enhancements

- **M10:** GPU acceleration for FFT/signal processing
- **M15:** Distributed computing for massive Monte Carlo
- **M20:** Machine learning model extraction (requires fast inference)

## References

- [Python Performance Tips](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)
- [NetworkX Documentation](https://networkx.org/)
- [pytest-benchmark](https://pytest-benchmark.readthedocs.io/)
- [py-spy](https://github.com/benfred/py-spy)
- [memory_profiler](https://pypi.org/project/memory-profiler/)
