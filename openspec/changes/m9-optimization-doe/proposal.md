# M9 - Optimization, Sensitivities, and Design of Experiments

## Why
Circuit designers need to tune parameters to meet specifications (e.g., phase margin > 60°, ENOB > 10 bits). SpiceLab currently lacks optimization tools to close the "simulate → measure → optimize" loop. Additionally, sensitivity analysis and Design of Experiments (DOE) are critical for understanding parameter influence and exploring design space efficiently.

## What Changes
- Add `spicelab/analysis/optimize.py` with wrappers for SciPy optimizers (Nelder-Mead, Powell, L-BFGS-B, differential evolution)
- Add optional support for advanced optimizers (Nevergrad, Optuna) via plugins
- Implement `spicelab/analysis/sensitivity.py` for local sensitivity via finite differences
- Add support for `.sens` analysis in Xyce with adapter for reading sensitivity results
- Implement `spicelab/analysis/doe.py` with DOE plans: Full Factorial, Latin Hypercube, Sobol sequences
- Integrate optimization with orchestrator for caching and parallel evaluation
- Add optimization history tracking (parameters, measurements, loss) with export to DataFrame

## Impact
- **Affected specs**: optimization, sensitivity, doe, orchestrator
- **Affected code**:
  - New: `spicelab/analysis/optimize.py`, `spicelab/analysis/sensitivity.py`, `spicelab/analysis/doe.py`
  - Modified: `spicelab/orchestrator.py` (optimization job scheduling)
  - Modified: `spicelab/engines/xyce_cli.py` (add .sens support)
  - New: `tests/test_optimization.py`, `tests/test_sensitivity.py`, `tests/test_doe.py`
  - New: `examples/optimization/`
- **Dependencies**: SciPy (required), Nevergrad/Optuna (optional)
