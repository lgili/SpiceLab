# M1: Stability & Testing Overhaul - Tasks

**Status:** Proposed
**Start Date:** TBD
**Target Completion:** TBD (4-6 weeks)

## Task Breakdown

### Phase 1: Foundation (Week 1)
- [ ] Create `tests/property/` directory structure
  - [ ] Add `__init__.py` and `conftest.py`
  - [ ] Create subdirectories: `test_components/`, `test_circuits/`, `test_analysis/`
- [ ] Add Hypothesis to `pyproject.toml` dev dependencies
  - [ ] Configure Hypothesis profiles (dev, CI, exhaustive)
- [ ] Setup pytest markers in `pytest.ini`
  - [ ] Add `@pytest.mark.property` marker
  - [ ] Add `@pytest.mark.slow` marker
  - [ ] Add `@pytest.mark.stress` marker
- [ ] Write 10 basic property tests
  - [ ] Components: Resistor, Capacitor, Inductor value ranges
  - [ ] Net: Name validation
  - [ ] Circuit: Basic add/connect operations

**Estimated Time:** 1 week
**Assignee:** TBD

---

### Phase 2: Property Testing (Week 2)
- [ ] Implement stateful circuit testing
  - [ ] Create `CircuitStateMachine` class
  - [ ] Add rules: add_component, connect_nodes, disconnect
  - [ ] Add invariants: no orphaned components, GND always connected
- [ ] Property tests for analysis specs
  - [ ] Valid parameter ranges (tstep < tstop, etc.)
  - [ ] Unit normalization idempotent
- [ ] Netlist parser roundtrip tests
  - [ ] Parse → regenerate → compare
  - [ ] Test with random valid netlists
- [ ] Configure Hypothesis settings
  - [ ] `dev` profile: max_examples=100, deadline=200ms
  - [ ] `ci` profile: max_examples=1000, deadline=1000ms
  - [ ] `exhaustive` profile: max_examples=10000, deadline=None

**Estimated Time:** 1 week
**Assignee:** TBD

---

### Phase 3: Fuzzing & Stress (Week 3)
- [ ] Write fuzzing tests (20+ targets)
  - [ ] Netlist parser (binary data, malformed syntax)
  - [ ] Value parser (arbitrary strings)
  - [ ] Component creation (invalid refs, values)
  - [ ] SPICE card generation
  - [ ] RAW file reader (corrupted files)
- [ ] Implement stress tests
  - [ ] Create `tests/stress/` directory
  - [ ] Test 100 component circuit (baseline)
  - [ ] Test 1,000 component circuit
  - [ ] Test 5,000 component circuit
  - [ ] Test 10,000 component circuit
  - [ ] Add timeout guards (fail if >1s for 10k)
- [ ] Memory profiling tests
  - [ ] Monte Carlo memory leak test (1k runs)
  - [ ] Large result file memory usage
  - [ ] Circuit object memory overhead
- [ ] Performance benchmarks
  - [ ] Netlist generation (various sizes)
  - [ ] Result parsing (RAW files)
  - [ ] Circuit connectivity analysis

**Estimated Time:** 1 week
**Assignee:** TBD

---

### Phase 4: Error Handling (Week 4)
- [ ] Define exception hierarchy
  - [ ] Create `spicelab/exceptions.py`
  - [ ] Define base `SpiceLabError`
  - [ ] Define specific exceptions (CircuitError, SimulationError, etc.)
  - [ ] Add docstrings and usage examples
- [ ] Replace assertions in `core/`
  - [ ] `circuit.py`: Replace assertions with CircuitError
  - [ ] `components.py`: Add value validation errors
  - [ ] `net.py`: Validate net names
  - [ ] `types.py`: Analysis parameter validation
- [ ] Replace assertions in `engines/`
  - [ ] `ngspice.py`: Handle missing binary, stderr parsing
  - [ ] `ltspice.py`: LTspice-specific errors
  - [ ] `xyce.py`: Xyce-specific errors
  - [ ] `base.py`: Generic simulation errors
- [ ] Add error recovery paths
  - [ ] Partial netlist export on error
  - [ ] Graceful engine fallback (try next available)
  - [ ] Result file partial reading (corrupted files)

**Estimated Time:** 1 week
**Assignee:** TBD

---

### Phase 5: Regression Suite (Week 5)
- [ ] Create `tests/regression/` structure
  - [ ] Add `conftest.py` with regression utilities
  - [ ] Setup naming convention: `issue_{number}_{description}.py`
- [ ] Document existing known bugs
  - [ ] Review GitHub issues
  - [ ] Review TODOs in codebase
  - [ ] Create regression tests for each
- [ ] Add regression test template
  - [ ] Docstring with issue link
  - [ ] Minimal reproduction case
  - [ ] Expected behavior assertion
- [ ] Update issue template
  - [ ] Add "Regression Test" checkbox
  - [ ] Require minimal reproduction code

**Estimated Time:** 1 week
**Assignee:** TBD

---

### Phase 6: CI & Coverage (Week 6)
- [ ] Configure GitHub Actions matrix
  - [ ] Create `.github/workflows/test.yml`
  - [ ] Matrix: 3 OSes (Ubuntu, macOS, Windows)
  - [ ] Matrix: 3 Python versions (3.10, 3.11, 3.12)
  - [ ] Install NGSpice in CI (for integration tests)
- [ ] Setup Codecov integration
  - [ ] Add Codecov token to GitHub secrets
  - [ ] Configure `.codecov.yml`
  - [ ] Add coverage upload step
  - [ ] Add badge to README
- [ ] Add coverage gates
  - [ ] Fail if overall coverage <90%
  - [ ] Fail if diff coverage <95%
  - [ ] Allow exemptions with `# pragma: no cover` + comment
- [ ] Generate coverage reports
  - [ ] HTML report (local development)
  - [ ] XML report (CI/Codecov)
  - [ ] Terminal summary (pytest-cov)
- [ ] Document coverage gaps
  - [ ] Identify uncovered code
  - [ ] Add justification comments
  - [ ] Create issues for hard-to-test areas

**Estimated Time:** 1 week
**Assignee:** TBD

---

## Acceptance Criteria

### Must Have
- [x] Overall branch coverage ≥95%
- [x] Core modules coverage ≥99%
- [x] 50+ property tests implemented
- [x] 20+ fuzzing tests implemented
- [x] 10+ stress tests (up to 10k components)
- [x] Exception hierarchy defined and used
- [x] Zero assertions in public API code
- [x] CI matrix passes on all platforms
- [x] Codecov integration active

### Should Have
- [ ] 20+ regression tests
- [ ] Performance benchmarks baseline
- [ ] Memory profiling for Monte Carlo
- [ ] Hypothesis seed database committed

### Nice to Have
- [ ] AFL fuzzing for future C/Rust extensions
- [ ] Parallel test execution (pytest-xdist)
- [ ] Coverage trending dashboard

## Testing Checklist

Before marking M1 as complete:
- [ ] Run full test suite locally (all markers)
- [ ] Verify CI passes on all 9 matrix combinations
- [ ] Check Codecov report shows ≥95% coverage
- [ ] Manually review uncovered lines (justified?)
- [ ] Run stress tests on production-like circuit
- [ ] Verify no memory leaks in Monte Carlo (1k runs)
- [ ] Test error messages are helpful (show to non-developer)

## Notes

- **Hypothesis Seed Database:** Commit `.hypothesis/` to git to preserve found edge cases
- **Slow Tests:** Mark tests >1s as `@pytest.mark.slow`, run separately in CI
- **Flaky Tests:** If a test is flaky, fix it immediately or remove it (never ignore)
- **Coverage Exemptions:** Use `# pragma: no cover` sparingly, always add comment explaining why

## Dependencies

- None (foundational milestone)

## Blocking

- M2 (Performance) depends on stress test infrastructure from M1
- M7 (Measurements) depends on property test patterns from M1

---

**Last Updated:** 2025-01-19
