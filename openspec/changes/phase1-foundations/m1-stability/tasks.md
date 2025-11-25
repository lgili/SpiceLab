# M1: Stability & Testing Overhaul - Tasks

**Status:** In Progress
**Start Date:** 2025-11-24
**Target Completion:** 2025-12-08 (2 weeks remaining)

## Task Breakdown

### Phase 1: Foundation (Week 1) ✅ COMPLETE
- [x] Create `tests/property/` directory structure
  - [x] Add `__init__.py` and `conftest.py`
  - [x] Create test files: `test_components_property.py`, `test_circuits_property.py`, `test_types_property.py`, `test_net_property.py`
- [x] Add Hypothesis to `pyproject.toml` dev dependencies
  - [x] Configure Hypothesis profiles (dev, CI, exhaustive)
- [x] Setup pytest markers in `pyproject.toml`
  - [x] Add `@pytest.mark.property` marker
  - [x] Add `@pytest.mark.slow` marker
  - [x] Add `@pytest.mark.stress` marker
  - [x] Add `@pytest.mark.regression` marker
  - [x] Add `@pytest.mark.benchmark` marker
- [x] Write 20+ basic property tests
  - [x] Components: Resistor, Capacitor, Inductor, Vdc value ranges
  - [x] Net: Name validation, hash, equality
  - [x] Port: Owner, name, role, hashability
  - [x] Circuit: Basic add/connect operations

**Completed:** 2025-11-24

---

### Phase 2: Property Testing (Week 2) ✅ COMPLETE
- [x] Implement stateful circuit testing
  - [x] Create `CircuitStateMachine` class in `test_circuits_property.py`
  - [x] Add rules: add_resistor, add_capacitor, add_inductor, add_voltage_source
  - [x] Add rules: connect_to_gnd, connect_to_named_net
  - [x] Add invariants: component_count_matches, can_generate_summary, can_generate_netlist_if_connected
- [x] Property tests for analysis specs
  - [x] Valid parameter ranges (mode, args validation)
  - [x] Unit normalization idempotent (SweepSpec)
  - [x] Probe creation and parsing (V(), I(), bare strings)
- [x] Property tests for types
  - [x] AnalysisSpec creation and roundtrip
  - [x] SweepSpec normalization
  - [x] Probe creation shortcuts
  - [x] ensure_* functions
  - [x] stable_hash determinism
  - [x] circuit_hash with extra context
- [x] Configure Hypothesis settings
  - [x] `dev` profile: max_examples=50, deadline=100ms
  - [x] `default` profile: max_examples=100, deadline=200ms
  - [x] `ci` profile: max_examples=1000, deadline=1000ms
  - [x] `exhaustive` profile: max_examples=10000, no deadline

**Completed:** 2025-11-24

---

### Phase 3: Fuzzing & Stress (Week 3) ✅ COMPLETE
- [x] Write stress tests (tests/stress/)
  - [x] Create `tests/stress/test_large_circuits.py`
  - [x] Test 100 component circuit (baseline)
  - [x] Test 500 component circuit
  - [x] Test 1,000 component circuit
  - [x] Test 2,000 component circuit
  - [x] Mixed component types (500 mixed R/L/C/V)
  - [x] Ladder network 100 stages
  - [x] Star topology 100 branches
  - [x] Mesh topology 10x10
  - [x] Scaling tests (netlist, hash)
  - [x] Memory tests
  - [x] Directive stress tests
  - [x] Summary/DOT generation stress tests
- [x] Performance benchmarks (tests/benchmarks/)
  - [x] Create `tests/benchmarks/test_circuit_benchmarks.py`
  - [x] Circuit creation benchmarks
  - [x] Component creation benchmarks
  - [x] Connection benchmarks
  - [x] Netlist generation benchmarks (10, 100, 500 components)
  - [x] Hash generation benchmarks
  - [x] Summary/DOT benchmarks
  - [x] Type creation benchmarks (AnalysisSpec, SweepSpec, Probe)
  - [x] Mixed component benchmarks
  - [x] Directive benchmarks

**Completed:** 2025-11-24

---

### Phase 4: Error Handling (Week 4) ✅ COMPLETE (Pre-existing)
- [x] Define exception hierarchy
  - [x] `spicelab/exceptions.py` exists with full hierarchy
  - [x] Base `SpiceLabError` defined
  - [x] Circuit errors: `CircuitError`, `FloatingNodeError`, `ShortCircuitError`, `InvalidConnectionError`, `ComponentNotFoundError`
  - [x] Simulation errors: `SimulationError`, `EngineNotFoundError`, `ConvergenceError`, `SimulationFailedError`, `AnalysisError`
  - [x] Parse errors: `ParseError`, `NetlistParseError`, `ResultParseError`, `ModelParseError`
  - [x] Validation errors: `ValidationError`, `ComponentValidationError`, `ParameterValidationError`, `CircuitValidationError`
  - [x] Configuration errors: `ConfigurationError`, `EngineConfigurationError`, `PathNotFoundError`
  - [x] All exceptions have docstrings, hints, and structured details

**Status:** Already complete from previous work

---

### Phase 5: Regression Suite (Week 5) ✅ COMPLETE
- [x] Create `tests/regression/` structure
  - [x] Add `__init__.py`
  - [x] Create `test_known_circuits.py` with regression tests
- [x] Implement regression tests
  - [x] RC Low-pass filter tests (netlist, hash, component count, summary)
  - [x] Voltage divider tests
  - [x] RLC series circuit tests
  - [x] Ladder network tests (3-stage)
  - [x] AnalysisSpec regression tests (op, tran, ac, dc, noise)
  - [x] Edge case tests (empty circuit, unconnected components, directives)
  - [x] Hash stability tests

**Completed:** 2025-11-24

---

### Phase 6: CI & Coverage (Week 6) ✅ COMPLETE
- [x] Configure GitHub Actions matrix
  - [x] Update `.github/workflows/ci.yml`
  - [x] Matrix: 3 OSes (Ubuntu, macOS, Windows)
  - [x] Matrix: 3 Python versions (3.10, 3.11, 3.12)
  - [x] Install NGSpice in CI (Linux and macOS)
  - [x] Add property-tests job
  - [x] Add regression-tests job
  - [x] Add stress-tests job
  - [x] Add benchmark-tests job
- [x] Setup Codecov integration
  - [x] Create `codecov.yml` configuration
  - [x] Add coverage upload step in CI
  - [x] Configure coverage thresholds (80% target)
- [x] Add pytest-timeout dependency
- [x] Add pytest-benchmark dependency

**Completed:** 2025-11-24

---

## Acceptance Criteria

### Must Have
- [x] Property tests directory structure created
- [x] 50+ property tests implemented (currently: ~80 tests across 4 files)
- [x] Stateful testing with CircuitStateMachine
- [x] Stress tests up to 2k components
- [x] Regression test suite for known circuits
- [x] Exception hierarchy defined (pre-existing, comprehensive)
- [x] CI matrix with multiple platforms (Ubuntu, macOS, Windows)
- [x] CI matrix with multiple Python versions (3.10, 3.11, 3.12)
- [x] Codecov integration configured
- [x] Benchmark tests with artifact upload

### Should Have
- [x] Performance benchmarks baseline (tests/benchmarks/)
- [ ] 20+ regression tests (currently ~15, need more)
- [ ] Memory profiling for Monte Carlo
- [ ] Hypothesis seed database committed

### Nice to Have
- [ ] AFL fuzzing for future C/Rust extensions
- [ ] Parallel test execution (pytest-xdist)
- [ ] Coverage trending dashboard

## Testing Checklist

Before marking M1 as complete:
- [ ] Run full test suite locally (all markers)
- [ ] Verify CI passes on all matrix combinations
- [ ] Check Codecov report shows ≥80% coverage
- [ ] Manually review uncovered lines (justified?)
- [ ] Run stress tests on production-like circuit
- [ ] Test error messages are helpful

## Files Created/Modified

### New Files
- `tests/property/test_circuits_property.py` - Circuit property tests + stateful testing
- `tests/property/test_types_property.py` - AnalysisSpec, SweepSpec, Probe, hash tests
- `tests/property/test_net_property.py` - Net and Port property tests
- `tests/stress/test_large_circuits.py` - Large circuit stress tests
- `tests/regression/test_known_circuits.py` - Known circuit regression tests
- `tests/benchmarks/test_circuit_benchmarks.py` - Performance benchmarks
- `codecov.yml` - Codecov configuration

### Modified Files
- `pyproject.toml` - Added regression marker, pytest-timeout dependency
- `.github/workflows/ci.yml` - Multi-platform matrix, Codecov, new test jobs

## Notes

- **Hypothesis Profiles:** Configured in both `pyproject.toml` and `tests/property/conftest.py`
- **Stress Tests:** Marked with `@pytest.mark.stress` and `@pytest.mark.slow`
- **Benchmarks:** Use pytest-benchmark, results uploaded as artifacts
- **Exception Hierarchy:** Already comprehensive from previous work

## Dependencies

- None (foundational milestone)

## Blocking

- M2 (Performance) depends on stress test infrastructure from M1
- M7 (Measurements) depends on property test patterns from M1

---

**Last Updated:** 2025-11-24
