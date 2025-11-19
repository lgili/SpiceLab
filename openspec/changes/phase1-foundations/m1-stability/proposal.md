# M1: Stability & Testing Overhaul

**Status:** Proposed
**Priority:** 🔴 CRITICAL
**Estimated Duration:** 4-6 weeks
**Dependencies:** None (foundational)

## Problem Statement

SpiceLab currently lacks comprehensive test coverage for edge cases, error scenarios, and stress conditions. This creates risk of silent failures in production usage and undermines user confidence.

### Current Gaps
- ❌ No property-based testing (edge cases undiscovered)
- ❌ No fuzzing (malformed netlists crash)
- ❌ No stress tests (circuits >1k components untested)
- ❌ ~60% test coverage (critical paths partially covered)
- ❌ Inconsistent error handling (some failures silent)
- ❌ No regression test suite

### Impact
- **Risk:** Production failures, data corruption, incorrect simulation results
- **User Experience:** Frustration from cryptic errors and unexpected crashes
- **Maintenance:** Hard to refactor confidently without comprehensive tests

## Objectives

1. Achieve **95%+ branch coverage** across all modules
2. Implement **property-based testing** with Hypothesis
3. Add **fuzzing** for netlist parser and input validation
4. Create **stress tests** for large circuits (1k, 5k, 10k components)
5. Establish **regression test suite** to prevent future breakage
6. Implement **graceful error recovery** (no silent failures)
7. Setup **CI matrix** across Python versions and platforms

## Technical Design

### 1. Property-Based Testing with Hypothesis

**Strategy:** Use Hypothesis to generate random valid/invalid inputs and discover edge cases.

#### Test Strategies
```python
from hypothesis import given, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule

# Test 1: Component values
@given(resistance=st.floats(min_value=1e-15, max_value=1e15, allow_nan=False))
def test_resistor_accepts_valid_resistance(resistance):
    """Resistor should accept any physically reasonable resistance."""
    r = Resistor("R1", resistance)
    assert r.resistance == resistance
    card = r.spice_card()
    assert "R1" in card

# Test 2: Circuit construction (stateful)
class CircuitStateMachine(RuleBasedStateMachine):
    """Stateful testing: build circuits with random operations."""

    def __init__(self):
        super().__init__()
        self.circuit = Circuit("test")
        self.components = []
        self.nets = [GND]

    @rule(comp_type=st.sampled_from([Resistor, Capacitor, Inductor]))
    def add_component(self, comp_type):
        ref = f"{comp_type.__name__[0]}{len(self.components)}"
        comp = comp_type(ref, 1e3)  # Default value
        self.circuit.add(comp)
        self.components.append(comp)

    @rule(data=st.data())
    def connect_random_nodes(self, data):
        if len(self.components) < 2:
            return
        c1 = data.draw(st.sampled_from(self.components))
        c2 = data.draw(st.sampled_from(self.components))
        net = Net(f"n{len(self.nets)}")
        self.nets.append(net)
        self.circuit.connect(c1.ports[0], net)
        self.circuit.connect(c2.ports[0], net)

    @rule()
    def validate_circuit(self):
        """Circuit should always be in valid state."""
        # Should not crash
        summary = self.circuit.summary()
        assert summary is not None

# Test 3: Netlist parsing roundtrip
@given(st.text(alphabet="RCL0-9 \n", min_size=10, max_size=1000))
def test_netlist_parser_never_crashes(netlist_text):
    """Parser should never crash, even on garbage input."""
    try:
        circuit = Circuit.from_netlist(netlist_text)
        # If it parsed, should be able to regenerate
        regenerated = circuit.build_netlist()
        assert isinstance(regenerated, str)
    except ValueError:
        # Expected for invalid netlists
        pass
```

#### Coverage Targets
- Component value ranges (extremes: 1e-15 to 1e15)
- Net connectivity (floating, shorted, multi-connected)
- Analysis parameters (zero, negative, extreme values)
- Netlist variations (different orderings, whitespace, comments)

### 2. Fuzzing with Hypothesis + Custom Fuzzer

**Strategy:** Generate malformed inputs to discover crash bugs.

#### Fuzzing Targets
```python
# Target 1: Netlist parser
@given(st.binary(min_size=0, max_size=10000))
def test_netlist_parser_fuzzing(binary_data):
    """Parser should handle arbitrary binary data gracefully."""
    try:
        text = binary_data.decode('utf-8', errors='ignore')
        Circuit.from_netlist(text)
    except (ValueError, ParseError):
        pass  # Expected
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")

# Target 2: SPICE value parsing
@given(st.text(min_size=1, max_size=50))
def test_value_parser_fuzzing(value_str):
    """Value parser should never crash."""
    from spicelab.core.units import parse_value_flexible
    try:
        result = parse_value_flexible(value_str)
        assert isinstance(result, (float, int))
    except ValueError:
        pass  # Expected for invalid values

# Target 3: Component instantiation
@given(
    ref=st.text(min_size=1, max_size=100),
    value=st.one_of(st.floats(), st.text(), st.integers(), st.none())
)
def test_component_creation_fuzzing(ref, value):
    """Component creation should validate inputs."""
    try:
        r = Resistor(ref, value)
        assert r.ref is not None
    except (ValueError, TypeError, ValidationError):
        pass  # Expected
```

#### AFL/LibFuzzer Integration (Optional - Phase 2)
For C extension modules (future Rust extensions):
```bash
# Use cargo-fuzz for Rust netlist parser
cargo fuzz run netlist_parser -- -max_total_time=300
```

### 3. Stress Tests for Large Circuits

**Strategy:** Test performance and correctness at scale.

#### Test Scenarios
```python
import pytest

@pytest.mark.slow
@pytest.mark.parametrize("n_components", [100, 1000, 5000, 10000])
def test_large_circuit_netlist_generation(n_components):
    """Test netlist generation scales to large circuits."""
    circuit = Circuit(f"large_{n_components}")

    # Build ladder network (R-C-R-C-...)
    prev_net = GND
    for i in range(n_components // 2):
        r = Resistor(f"R{i}", "1k")
        c = Capacitor(f"C{i}", "100n")
        next_net = Net(f"n{i}")

        circuit.add(r)
        circuit.add(c)
        circuit.connect(r.ports[0], prev_net)
        circuit.connect(r.ports[1], next_net)
        circuit.connect(c.ports[0], next_net)
        circuit.connect(c.ports[1], GND)
        prev_net = next_net

    # Should complete in <1 second for 10k components
    import time
    start = time.perf_counter()
    netlist = circuit.build_netlist()
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"Netlist generation took {elapsed:.2f}s (target: <1s)"
    assert len(netlist.split('\n')) > n_components

@pytest.mark.slow
@pytest.mark.engine
def test_large_circuit_simulation(tmp_path):
    """Test simulation of large circuits completes successfully."""
    circuit = _build_large_circuit(n=1000)

    tran = AnalysisSpec("tran", {"tstep": "1us", "tstop": "1ms"})
    handle = run_simulation(circuit, [tran], engine="ngspice")

    ds = handle.dataset()
    assert len(ds.data_vars) > 100  # Many nodes
    assert ds.sizes["time"] > 100   # Sufficient time points

@pytest.mark.slow
def test_memory_usage_large_monte_carlo(benchmark):
    """Monte Carlo should not leak memory."""
    import tracemalloc

    circuit = _build_rc_circuit()
    r1 = circuit.get_component("R1")

    tracemalloc.start()

    mc = monte_carlo(
        circuit=circuit,
        mapping={r1: NormalPct(0.05)},
        n=1000,
        analyses=[AnalysisSpec("op", {})],
        engine="ngspice",
    )

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Should use <500MB for 1k runs
    assert peak < 500 * 1024 * 1024, f"Peak memory: {peak / 1e6:.1f}MB"
```

### 4. Regression Test Suite

**Strategy:** Capture known bugs and prevent regressions.

#### Structure
```
tests/regression/
├── conftest.py
├── issue_001_floating_node.py
├── issue_002_unicode_ref.py
├── issue_003_negative_capacitance.py
└── ...
```

#### Example Regression Test
```python
# tests/regression/issue_001_floating_node.py
"""
Regression test for Issue #1: Floating nodes not detected.

Background: Circuits with disconnected components would silently fail
or produce incorrect results. Now should raise ValidationError.
"""
import pytest
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor
from spicelab.core.net import GND, Net
from spicelab.validators import validate_circuit

def test_floating_node_detected():
    """Floating nodes should be detected during validation."""
    circuit = Circuit("floating")

    # Create disconnected component
    r1 = Resistor("R1", "1k")
    circuit.add(r1)
    # Oops - forgot to connect!

    with pytest.raises(ValidationError, match="floating"):
        validate_circuit(circuit, strict=True)
```

### 5. Graceful Error Recovery

**Strategy:** Replace assertions with explicit error handling.

#### Error Hierarchy
```python
# spicelab/exceptions.py

class SpiceLabError(Exception):
    """Base exception for all SpiceLab errors."""
    pass

class CircuitError(SpiceLabError):
    """Errors related to circuit construction."""
    pass

class FloatingNodeError(CircuitError):
    """Circuit contains disconnected nodes."""
    def __init__(self, nodes: list[Net]):
        self.nodes = nodes
        super().__init__(f"Floating nodes detected: {nodes}")

class ShortCircuitError(CircuitError):
    """Circuit contains short circuit."""
    pass

class SimulationError(SpiceLabError):
    """Errors during simulation execution."""
    pass

class EngineNotFoundError(SimulationError):
    """SPICE engine binary not found."""
    def __init__(self, engine: str, path: str | None = None):
        self.engine = engine
        self.path = path
        msg = f"Engine '{engine}' not found"
        if path:
            msg += f" at '{path}'"
        super().__init__(msg)

class ConvergenceError(SimulationError):
    """Simulation failed to converge."""
    pass

class ParseError(SpiceLabError):
    """Errors parsing netlists or result files."""
    pass
```

#### Error Handling Pattern
```python
# Before (crashes on error)
def build_netlist(self) -> str:
    assert self.components, "No components"
    netlist = "* Circuit\n"
    for comp in self.components:
        netlist += comp.spice_card() + "\n"
    return netlist

# After (graceful error)
def build_netlist(self) -> str:
    if not self.components:
        raise CircuitError("Cannot build netlist: circuit has no components")

    try:
        lines = ["* Circuit netlist"]
        for comp in self.components:
            card = comp.spice_card()
            if not card:
                raise CircuitError(f"Component {comp.ref} generated empty SPICE card")
            lines.append(card)
        return "\n".join(lines)
    except Exception as e:
        raise CircuitError(f"Failed to build netlist: {e}") from e
```

### 6. CI Matrix

**Strategy:** Test across Python versions and platforms.

#### GitHub Actions Configuration
```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v1

      - name: Set up Python ${{ matrix.python-version }}
        run: uv python install ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          uv sync --all-extras --dev

      - name: Run unit tests
        run: |
          uv run pytest tests/unit -v --cov=spicelab --cov-report=xml

      - name: Run property tests
        run: |
          uv run pytest tests/property -v --hypothesis-profile=ci

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: ${{ matrix.os }}-py${{ matrix.python-version }}

  stress-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v1
      - name: Run stress tests
        run: |
          uv run pytest tests/stress -v --tb=short
```

## Implementation Plan

### Week 1: Foundation
- [ ] Create `tests/property/` directory structure
- [ ] Add Hypothesis to dev dependencies
- [ ] Write 10 basic property tests (components, nets, circuits)
- [ ] Setup pytest markers (`@pytest.mark.property`, `@pytest.mark.slow`)

### Week 2: Property Testing
- [ ] Implement stateful circuit testing (RuleBasedStateMachine)
- [ ] Add property tests for analysis specs
- [ ] Property tests for netlist parser roundtrip
- [ ] Add Hypothesis profiles (dev, CI, exhaustive)

### Week 3: Fuzzing & Stress
- [ ] Write fuzzing tests for parser (20+ targets)
- [ ] Implement stress tests (100, 1k, 5k, 10k components)
- [ ] Memory profiling for Monte Carlo
- [ ] Performance benchmarks (track regressions)

### Week 4: Error Handling
- [ ] Define exception hierarchy (`exceptions.py`)
- [ ] Replace assertions with explicit errors (core/)
- [ ] Replace assertions with explicit errors (engines/)
- [ ] Add error recovery paths (e.g., partial netlist export on failure)

### Week 5: Regression Suite
- [ ] Create `tests/regression/` structure
- [ ] Document known bugs as regression tests
- [ ] Add regression tests for reported issues
- [ ] Setup issue template with regression test requirement

### Week 6: CI & Coverage
- [ ] Configure GitHub Actions matrix (3 OSes × 3 Python versions)
- [ ] Setup Codecov integration
- [ ] Add coverage gates (fail if <90%)
- [ ] Generate coverage reports in CI
- [ ] Document uncovered code and justification

## Success Metrics

### Coverage
- [ ] Overall branch coverage: **95%+**
- [ ] Core modules (circuit, components, net): **99%+**
- [ ] Engines: **90%+**
- [ ] Analysis: **95%+**

### Test Counts
- [ ] Property tests: **50+**
- [ ] Fuzzing tests: **20+**
- [ ] Stress tests: **10+**
- [ ] Regression tests: **20+** (grows with bug reports)
- [ ] Total tests: **500+** (from ~150 currently)

### Quality Gates
- [ ] Zero uncaught exceptions in CI
- [ ] Zero silent failures (all errors logged or raised)
- [ ] All tests pass on 3 OSes × 3 Python versions
- [ ] Hypothesis finds <5 bugs per week (plateau)

### Performance
- [ ] Stress test (10k components): <1s netlist generation
- [ ] Monte Carlo (1k runs): <500MB peak memory
- [ ] Test suite runtime: <5 minutes (unit + property)

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Test suite too slow | Dev velocity | Split into fast/slow markers, run slow in nightly |
| Hypothesis finds too many bugs | Overwhelm | Prioritize, fix critical first, add to backlog |
| Coverage impossible in some areas | Can't hit 95% | Document exemptions, use `# pragma: no cover` sparingly |
| Flaky tests (timing-dependent) | CI unreliable | Use deterministic timeouts, retry flaky tests |
| Platform-specific failures | Debugging hard | Add detailed logging, use remote debugging |

## Dependencies

**Required:**
- Hypothesis (`pip install hypothesis`)
- pytest-benchmark
- tracemalloc (stdlib)
- Codecov account (free for open source)

**Optional:**
- AFL/LibFuzzer (for future C/Rust extensions)
- pytest-xdist (parallel test execution)

## Future Enhancements

- **M2:** Integrate stress tests with performance benchmarks
- **M7:** Add measurement-specific property tests
- **M12:** Plugin system property tests (invariants)
- **Continuous:** Regression suite grows with every bug report

## References

- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [Property-Based Testing Patterns](https://increment.com/testing/in-praise-of-property-based-testing/)
- [AFL Fuzzing](https://github.com/google/AFL)
- [pytest-benchmark](https://pytest-benchmark.readthedocs.io/)
