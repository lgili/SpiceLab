## Why

The PT1000 signal conditioning circuit project revealed significant gaps in spicelab's tolerance analysis capabilities. Users currently need to write extensive boilerplate code for common industrial requirements like correlated parameter variation, multi-temperature sweeps, and process capability metrics (Cpk). These features are essential for production-ready circuit validation but are not directly supported by the library.

## What Changes

### 1. Correlated Parameter Groups
- Add `CorrelatedGroup` class to link parameters that vary together (same-lot components)
- Support mixing correlated and independent parameters in same analysis

### 2. Absolute Tolerance Distributions
- Add `NormalAbs(sigma_value)` for absolute tolerances (e.g., op-amp offset ±2mV)
- Complement existing `UniformAbs` with normal distribution equivalent

### 3. Temperature Sweep Integration
- Add `monte_carlo_sweep()` function for multi-condition MC analysis
- Add `wca_sweep()` function for multi-condition WCA analysis
- Return results with temperature as a dimension

### 4. Mixed Tolerance WCA
- Extend `run_wca()` to accept absolute tolerances alongside relative
- Support tuple syntax: `{R1: 0.001, Voff: (0.002, 'abs')}`

### 5. Simplified Result Extraction
- Add `result.get_voltage(node)` and `result.get_current(component)` methods
- Reduce boilerplate for common extraction patterns

### 6. Process Capability Metrics
- Add `mc_result.cpk(metric, spec)` method
- Add `mc_result.yield_estimate(spec)` method
- Add `mc_result.sigma_level()` method
- Integrate with existing `compute_stats()` infrastructure

### 7. Automated Report Generation
- Add `generate_report()` method for HTML/PDF output
- Include histograms, Cpk charts, worst corners, sensitivity analysis

## Impact

- Affected specs: New capability `tolerance-analysis`
- Affected code:
  - `spicelab/analysis/montecarlo.py` - Add correlated groups, NormalAbs
  - `spicelab/analysis/wca.py` - Add absolute tolerance support
  - `spicelab/analysis/stats.py` - Add Cpk/yield methods
  - `spicelab/analysis/sweep.py` - New file for temperature sweep integration
  - `spicelab/analysis/report.py` - New file for report generation
  - `spicelab/core/results.py` - Add simplified extraction methods
