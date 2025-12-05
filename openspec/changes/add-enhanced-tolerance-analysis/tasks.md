## 1. Correlated Parameter Groups

- [x] 1.1 Create `CorrelatedGroup` class in `spicelab/analysis/montecarlo.py`
- [x] 1.2 Implement group sampling logic (single random factor for all members)
- [x] 1.3 Update `monte_carlo()` to accept mixed dict with groups and individual params
- [x] 1.4 Add unit tests for correlated sampling
- [x] 1.5 Add integration test comparing correlated vs independent results

## 2. Absolute Tolerance Distributions

- [x] 2.1 Add `NormalAbs(sigma_value)` distribution class
- [x] 2.2 Add `TriangularAbs(half_range)` distribution class
- [x] 2.3 Update distribution base class to support absolute mode
- [x] 2.4 Add unit tests for absolute distributions
- [x] 2.5 Update documentation with examples (in docs/tolerance-analysis.md)

## 3. Temperature Sweep Integration

- [x] 3.1 Create `spicelab/analysis/sweep.py` module
- [x] 3.2 Implement `monte_carlo_sweep(circuit, tolerances, conditions, n_iterations)`
- [x] 3.3 Implement `wca_sweep(circuit, tolerances, conditions)`
- [x] 3.4 Return `MultiConditionResult` with condition as dimension
- [x] 3.5 Add DataFrame export with multi-index (condition, iteration)
- [x] 3.6 Add unit tests for sweep functions
- [x] 3.7 Add example demonstrating temperature sweep (in docs/tolerance-analysis.md)

## 4. Mixed Tolerance WCA

- [x] 4.1 Extend `run_wca()` signature to accept `(value, 'abs')` tuples
- [x] 4.2 Implement corner generation for absolute tolerances
- [x] 4.3 Update WCA result display to show tolerance types
- [x] 4.4 Add unit tests for mixed tolerance WCA
- [x] 4.5 Backward compatibility test with existing API

## 5. Simplified Result Extraction

- [x] 5.1 Add `SimulationResult.get_voltage(node: str) -> float` method
- [x] 5.2 Add `SimulationResult.get_current(component: str) -> float` method
- [x] 5.3 Add `SimulationResult.get_power(component: str) -> float` method
- [x] 5.4 Handle case-insensitive node names
- [x] 5.5 Add helpful error messages for missing nodes
- [x] 5.6 Add unit tests for extraction methods

## 6. Process Capability Metrics

- [x] 6.1 Add `MonteCarloResult.cpk(metric, lsl, usl)` method
- [x] 6.2 Add `MonteCarloResult.yield_estimate(lsl, usl)` method
- [x] 6.3 Add `MonteCarloResult.sigma_level(lsl, usl)` method
- [x] 6.4 Add `MonteCarloResult.process_summary(metric, spec)` for full report
- [x] 6.5 Integrate with existing `compute_stats()` infrastructure
- [x] 6.6 Add unit tests with known Cpk values
- [x] 6.7 Add documentation with industry-standard definitions (in tolerance-analysis.md)

## 7. Automated Report Generation

- [x] 7.1 Create `spicelab/analysis/report.py` module
- [x] 7.2 Implement `generate_report(result, output_path, format='html')`
- [x] 7.3 Add histogram section with spec limits overlay
- [x] 7.4 Add Cpk/yield summary table
- [x] 7.5 Add worst-case corners table
- [ ] 7.6 Add sensitivity analysis section (deferred - optional feature)
- [x] 7.7 Support HTML output with embedded plots
- [ ] 7.8 Support PDF output (deferred - optional, requires weasyprint)
- [x] 7.9 Add unit tests for report generation
- [x] 7.10 Add example in documentation (in monte-carlo.md)

## 8. Documentation and Examples

- [x] 8.1 Update API reference for new classes and methods
- [x] 8.2 Create tutorial: "Production Tolerance Analysis" (docs/tolerance-analysis.md)
- [x] 8.3 Add example: PT1000 circuit using new API (in tolerance-analysis.md)
- [x] 8.4 Update README with new capabilities
- [x] 8.5 Add migration guide from manual approach to new API
