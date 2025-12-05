## Context

This enhancement addresses gaps identified during the PT1000 signal conditioning circuit validation project. The current spicelab Monte Carlo and WCA implementations work well for basic analysis but require significant boilerplate for production-grade tolerance analysis used in industrial applications.

**Stakeholders:**
- Circuit designers performing production validation
- Quality engineers calculating process capability
- Test engineers generating compliance reports

**Constraints:**
- Must maintain backward compatibility with existing API
- Must not break existing tests or notebooks
- Performance should remain acceptable for 10k+ iterations

## Goals / Non-Goals

### Goals
- Reduce boilerplate code for common tolerance analysis patterns by 70%+
- Support industry-standard process capability metrics (Cpk, yield, sigma level)
- Enable realistic modeling of same-lot component correlation
- Provide one-command report generation for documentation

### Non-Goals
- Full statistical process control (SPC) charting
- Real-time production monitoring
- Component supplier qualification workflows
- HALT/HASS stress testing automation

## Decisions

### Decision 1: Correlated Groups via Wrapper Class

**What:** Create `CorrelatedGroup` class that wraps multiple components with shared random factor.

**Why:**
- Clean separation from independent variation
- Explicit in API (user sees correlation)
- Easy to mix with independent params

**Alternative considered:** Matrix-based correlation coefficients
- Rejected: Over-engineered for common use case (same-lot = perfect correlation)

```python
# Chosen approach
monte_carlo(circuit, {
    CorrelatedGroup([R1, R2, R3], NormalPct(0.1)): None,  # Same batch
    Voff1: NormalAbs(0.002),  # Independent
    Voff2: NormalAbs(0.002),  # Independent
})

# Alternative (rejected)
monte_carlo(circuit, params, correlation_matrix=[[1, 0.99, 0.99], [0.99, 1, 0.99], ...])
```

### Decision 2: Absolute Distributions as Separate Classes

**What:** Add `NormalAbs`, `TriangularAbs` alongside existing `UniformAbs`.

**Why:**
- Consistent with existing pattern (`UniformAbs` already exists)
- Clear intent in code
- No magic mode flags

**Alternative considered:** Add `absolute=True` flag to existing classes
- Rejected: Changes meaning of `sigma` parameter, confusing

```python
# Chosen approach
NormalAbs(0.002)  # ±2mV, 3-sigma

# Alternative (rejected)
NormalPct(0.002, absolute=True)  # Confusing: is 0.002 a percentage or value?
```

### Decision 3: Sweep Functions Return MultiConditionResult

**What:** New `monte_carlo_sweep()` returns results indexed by condition.

**Why:**
- Natural extension of existing API
- DataFrame export with multi-index is intuitive
- Enables cross-condition analysis (worst temp, etc.)

```python
result = monte_carlo_sweep(circuit, tolerances,
    conditions={'temperature': [-40, 25, 85]},
    n_iterations=1000
)

# Access by condition
result.at_condition(temperature=25).cpk(...)

# Or as DataFrame
df = result.to_dataframe()  # Multi-index: (temperature, iteration)
```

### Decision 4: Tuple Syntax for Mixed WCA Tolerances

**What:** Accept `(value, 'abs')` tuples in WCA tolerance dict.

**Why:**
- Minimal API change
- Backward compatible (plain float = relative)
- Explicit about tolerance type

```python
run_wca(circuit, {
    R1: 0.001,           # 0.1% relative (existing)
    Voff: (0.002, 'abs') # 2mV absolute (new)
})
```

### Decision 5: Result Extraction as Instance Methods

**What:** Add `get_voltage()`, `get_current()` to `SimulationResult`.

**Why:**
- Discoverable via autocomplete
- Handles dataset API internally
- Provides helpful errors

**Alternative considered:** Standalone functions
- Rejected: Less discoverable, requires importing extra module

```python
# Chosen approach
result.get_voltage('vout')

# Alternative (rejected)
from spicelab.analysis.extract import get_voltage
get_voltage(result, 'vout')
```

## Risks / Trade-offs

### Risk 1: API Complexity Growth
- **Risk:** Too many options makes API harder to learn
- **Mitigation:** Clear defaults, progressive disclosure (basic → advanced)

### Risk 2: Performance with Correlation
- **Risk:** Correlated sampling adds overhead
- **Mitigation:** Benchmark shows <5% overhead for grouping logic

### Risk 3: Report Generation Dependencies
- **Risk:** HTML/PDF generation adds dependencies
- **Mitigation:** Make weasyprint optional, HTML works standalone

## Migration Plan

### Phase 1: Core Features (Sprint 1-2)
1. Add `NormalAbs`, `TriangularAbs` distributions
2. Add `CorrelatedGroup` class
3. Add mixed tolerance WCA support

### Phase 2: Integration (Sprint 2-3)
4. Add sweep functions
5. Add result extraction methods
6. Add Cpk/yield methods

### Phase 3: Reporting (Sprint 3-4)
7. Add report generation
8. Update documentation
9. Create example notebooks

### Rollback
- All features are additive (no breaking changes)
- Can disable features via feature flags if issues arise
- Existing API remains unchanged

## Open Questions

1. Should `CorrelatedGroup` support partial correlation (e.g., 0.8 instead of 1.0)?
   - Current decision: No, keep simple. Full correlation or independent.

2. Should report generation support LaTeX output for academic papers?
   - Current decision: Defer. HTML and PDF cover most use cases.

3. Should sweep support arbitrary conditions beyond temperature?
   - Current decision: Yes, generic `conditions` dict allows any parameter.
