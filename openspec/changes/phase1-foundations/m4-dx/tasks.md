# M4: Developer Experience (DX) - Tasks

**Status:** Proposed
**Start Date:** TBD
**Target Completion:** TBD (4-6 weeks)
**Dependencies:** M1 (error handling), M3 (component library)

## Task Breakdown

### Phase 1: Error Messages & Suggestions (Week 1)
- [ ] Design error hierarchy
  - [ ] Create `spicelab/errors.py` with base classes
  - [ ] Implement CircuitError with rich formatting
  - [ ] Add message, hint, code_example, docs_url fields
  - [ ] Implement __str__ with rich.console formatting
- [ ] Create specific error classes
  - [ ] EmptyCircuitError
  - [ ] FloatingNodeError
  - [ ] ComponentNotFoundError
  - [ ] InvalidValueError
  - [ ] ConvergenceError (enhanced)
  - [ ] 10+ domain-specific errors
- [ ] Implement suggestion engine
  - [ ] Create `spicelab/validation/suggestions.py`
  - [ ] Add fuzzy matching for component types
  - [ ] Add value correction suggestions
  - [ ] Handle common typos (1kO -> 1k, 10uF -> 10u)
- [ ] Replace assertions
  - [ ] Replace in circuit.py
  - [ ] Replace in components.py
  - [ ] Replace in engines/
  - [ ] Replace in analysis.py
- [ ] Test error messages
  - [ ] Create test suite for all error types
  - [ ] Validate formatting in terminal
  - [ ] Get user feedback (5+ developers)

**Estimated Time:** 1 week

---

### Phase 2: Validation & Diff (Week 2)
- [ ] Implement circuit validator
  - [ ] Create `spicelab/validation/circuit_checks.py`
  - [ ] Implement CircuitValidator class
  - [ ] Add ValidationResult dataclass
  - [ ] Implement _check_floating_nodes()
  - [ ] Implement _check_voltage_source_loop()
  - [ ] Implement _check_unrealistic_values()
  - [ ] Implement _check_missing_ground()
  - [ ] Add severity levels (error, warning, info)
- [ ] Create netlist diff tool
  - [ ] Create `spicelab/visualization/netlist_diff.py`
  - [ ] Implement NetlistDiff class
  - [ ] Add unified diff mode
  - [ ] Add side-by-side comparison mode
  - [ ] Use rich for syntax highlighting
  - [ ] Add color coding (green=added, red=removed)
- [ ] Integrate validation
  - [ ] Add validate() method to Circuit
  - [ ] Run validation before simulation (optional flag)
  - [ ] Create CLI command: `spicelab validate circuit.py`
  - [ ] Add to pre-commit hook (optional)
- [ ] Write tests
  - [ ] Test all validation rules
  - [ ] Test diff visualization
  - [ ] Test false positive rate (<5%)

**Estimated Time:** 1 week

---

### Phase 3: Autocomplete & Type Stubs (Week 3)
- [ ] Create type stub files
  - [ ] Create .pyi files for all modules
  - [ ] Add Literal hints for common values
  - [ ] Document in docstrings: "Common values: 1k, 10k, 100k"
  - [ ] Test in VSCode (Python extension)
  - [ ] Test in PyCharm
- [ ] Add VSCode snippets
  - [ ] Create `.vscode/spicelab.code-snippets`
  - [ ] Add resistor snippets (R1k, R10k, R100k)
  - [ ] Add capacitor snippets (C100n, C1u, C10u)
  - [ ] Add circuit template snippets
  - [ ] Document in IDE setup guide
- [ ] Create autocomplete metadata
  - [ ] Add __all__ to all modules
  - [ ] Improve docstrings (examples in each)
  - [ ] Add parameter descriptions
  - [ ] Generate API reference
- [ ] Test IDE integration
  - [ ] Verify autocomplete works in VSCode
  - [ ] Verify autocomplete works in PyCharm
  - [ ] Test on fresh install
  - [ ] Document any setup steps needed

**Estimated Time:** 1 week

---

### Phase 4: Circuit Templates (Week 4)
- [ ] Create template package
  - [ ] Create `spicelab/templates/` package
  - [ ] Add subdirectories: filters/, amplifiers/, power/, oscillators/
- [ ] Implement filter templates
  - [ ] LowPassRCFilter (1st order)
  - [ ] HighPassRCFilter (1st order)
  - [ ] BandPassFilter (2nd order)
  - [ ] NotchFilter (twin-T)
  - [ ] Sallen-Key filters (optional)
- [ ] Implement amplifier templates
  - [ ] NonInvertingAmplifier (OpAmp)
  - [ ] InvertingAmplifier (OpAmp)
  - [ ] DifferentialAmplifier
  - [ ] InstrumentationAmplifier
  - [ ] CommonEmitterAmplifier (BJT)
- [ ] Implement power supply templates
  - [ ] LinearRegulator (LDO-based)
  - [ ] BuckConverter (switching)
  - [ ] BoostConverter (switching)
  - [ ] InverterCircuit (negative voltage)
- [ ] Implement oscillator templates
  - [ ] RCOscillator (relaxation)
  - [ ] CrystalOscillator
  - [ ] VCO (voltage-controlled)
  - [ ] Astable multivibrator
- [ ] Document all templates
  - [ ] Add comprehensive docstrings
  - [ ] Include design equations
  - [ ] Add example usage for each
  - [ ] Create tutorial: "Building with Templates"

**Estimated Time:** 1 week

---

### Phase 5: Interactive Troubleshooting (Week 5)
- [ ] Create troubleshooter
  - [ ] Create `spicelab/troubleshooting/interactive.py`
  - [ ] Implement InteractiveTroubleshooter class
  - [ ] Use rich.prompt for interactive questions
  - [ ] Add color-coded output (rich.console)
- [ ] Implement diagnostics
  - [ ] diagnose_convergence_failure()
  - [ ] diagnose_empty_results()
  - [ ] diagnose_simulation_timeout()
  - [ ] diagnose_unexpected_output()
- [ ] Add guided fixes
  - [ ] Suggest .OPTIONS adjustments
  - [ ] Suggest initial conditions
  - [ ] Suggest component value changes
  - [ ] Offer to apply fixes automatically
- [ ] Create CLI command
  - [ ] `spicelab troubleshoot circuit.py`
  - [ ] Interactive mode (questionnaire)
  - [ ] Non-interactive mode (auto-diagnose)
- [ ] Test with real scenarios
  - [ ] Test with 10+ common errors
  - [ ] Validate suggestions are helpful
  - [ ] Get user feedback

**Estimated Time:** 1 week

---

### Phase 6: Integration & UX Testing (Week 6)
- [ ] Integration
  - [ ] Integrate all DX features into main library
  - [ ] Add CLI commands (validate, troubleshoot, diff)
  - [ ] Update error handling throughout codebase
  - [ ] Ensure all features work together
- [ ] Create comprehensive examples
  - [ ] Example: error message showcase
  - [ ] Example: validation workflow
  - [ ] Example: using templates
  - [ ] Example: troubleshooting convergence
  - [ ] Add to examples/ directory
- [ ] User testing
  - [ ] Recruit 5+ developers (beginners + experienced)
  - [ ] Give debugging tasks (measure time)
  - [ ] Collect feedback (survey)
  - [ ] Measure debugging time reduction
- [ ] Refine based on feedback
  - [ ] Adjust error message verbosity
  - [ ] Fine-tune validation rules
  - [ ] Improve template APIs
  - [ ] Fix usability issues
- [ ] Update documentation
  - [ ] Write troubleshooting guide
  - [ ] Document all error types
  - [ ] Add IDE setup guide
  - [ ] Create template catalog reference

**Estimated Time:** 1 week

---

## Acceptance Criteria

### Must Have
- [ ] 100% of errors have helpful messages (no raw stack traces exposed to users)
- [ ] 80%+ of errors include correction suggestions
- [ ] Circuit validator catches 90%+ of common errors
- [ ] Netlist diff visualization works in terminal
- [ ] Autocomplete works in VSCode and PyCharm
- [ ] 10+ circuit templates implemented and documented
- [ ] Interactive troubleshooter handles convergence failures
- [ ] **60% reduction in average debugging time** (measured with 5+ users)

### Should Have
- [ ] VSCode code snippets for common components
- [ ] Validation runs in <100ms
- [ ] All templates have design equations documented
- [ ] Error messages tested with beginners

### Nice to Have
- [ ] VSCode extension (syntax highlighting, preview)
- [ ] Web-based circuit debugger
- [ ] AI-powered error suggestions (future M19)

## Testing Checklist

Before marking M4 as complete:
- [ ] All error classes tested (unit tests)
- [ ] Validation rules tested (no false positives)
- [ ] Netlist diff tested (various scenarios)
- [ ] Autocomplete verified in VSCode
- [ ] Autocomplete verified in PyCharm
- [ ] All templates tested (simulate correctly)
- [ ] Troubleshooter tested with real errors
- [ ] User testing completed (5+ developers)
- [ ] Debugging time measured (before/after)
- [ ] User satisfaction score >4.5/5

## Metrics Tracking

| Metric | Before M4 | Target | Measured |
|--------|-----------|--------|----------|
| Avg. debugging time | 30 min | 12 min | TBD |
| Error understanding | 60% | 95% | TBD |
| Validation coverage | 0% | 90% | TBD |
| Template usage | 0% | 50% | TBD |
| User satisfaction | 3.0/5 | 4.5/5 | TBD |

## Dependencies

- M1 (Stability) - error handling infrastructure
- M3 (Components) - library needed for autocomplete/templates

## Blocking

- M5 (Docs) - examples depend on templates
- M11 (UX) - web UI builds on DX foundations

---

**Last Updated:** 2025-01-19
