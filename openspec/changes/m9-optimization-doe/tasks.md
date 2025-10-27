# Implementation Tasks - M9 Optimization, Sensitivity, and DOE

## 1. Optimization Infrastructure
- [ ] 1.1 Create `spicelab/analysis/optimize.py`
- [ ] 1.2 Define `OptimizationSpec` Pydantic model (algorithm, bounds, constraints, options)
- [ ] 1.3 Implement `OptimizationResult` with best params, history, convergence info
- [ ] 1.4 Create objective function builder that chains: Circuit(params) → run → measure → loss

## 2. SciPy Optimizers Integration
- [ ] 2.1 Wrap `scipy.optimize.minimize` with method selection (Nelder-Mead, Powell, L-BFGS-B, COBYLA, SLSQP)
- [ ] 2.2 Wrap `scipy.optimize.differential_evolution` for global optimization
- [ ] 2.3 Wrap `scipy.optimize.basinhopping` for multi-modal problems
- [ ] 2.4 Implement bounds enforcement and constraint handling
- [ ] 2.5 Add callback for iteration tracking (log params, loss, step)

## 3. Advanced Optimizers (Optional)
- [ ] 3.1 Add Nevergrad integration (derivative-free, noisy objectives)
- [ ] 3.2 Add Optuna integration (Bayesian optimization, pruning)
- [ ] 3.3 Make advanced optimizers optional dependencies
- [ ] 3.4 Provide unified API regardless of backend

## 4. Sensitivity Analysis
- [ ] 4.1 Create `spicelab/analysis/sensitivity.py`
- [ ] 4.2 Implement local sensitivity by finite differences (forward, central)
- [ ] 4.3 Add sensitivity normalization (percentage change)
- [ ] 4.4 Support multi-dimensional sensitivity (Jacobian matrix)
- [ ] 4.5 Add sensitivity ranking by magnitude

## 5. Xyce Sensitivity Support
- [ ] 5.1 Extend Xyce engine to support `.sens` analysis
- [ ] 5.2 Parse Xyce sensitivity output files (.sens.prn)
- [ ] 5.3 Normalize sensitivity data to xarray.Dataset format
- [ ] 5.4 Map sensitivity derivatives to parameter names

## 6. Design of Experiments (DOE)
- [ ] 6.1 Create `spicelab/analysis/doe.py`
- [ ] 6.2 Implement Full Factorial design
- [ ] 6.3 Implement Latin Hypercube Sampling (LHS) using `scipy.stats.qmc.LatinHypercube`
- [ ] 6.4 Implement Sobol sequence using `scipy.stats.qmc.Sobol`
- [ ] 6.5 Add DOE result visualization (scatter matrix, Pareto fronts)

## 7. Optimization Integration with Orchestrator
- [ ] 7.1 Add optimization job type to orchestrator
- [ ] 7.2 Implement caching of simulation results during optimization (avoid re-running same params)
- [ ] 7.3 Add parallel objective function evaluation (population-based algorithms)
- [ ] 7.4 Track optimization history in structured format

## 8. Multi-Objective Optimization
- [ ] 8.1 Add support for multi-objective problems (Pareto fronts)
- [ ] 8.2 Implement weighted sum method for multi-objective
- [ ] 8.3 Add NSGA-II integration (via pymoo or similar)
- [ ] 8.4 Visualize Pareto fronts with trade-off curves

## 9. Optimization Utilities
- [ ] 9.1 Implement restart mechanisms for stuck optimizations
- [ ] 9.2 Add convergence detection and early stopping
- [ ] 9.3 Provide default loss functions (L1, L2, max error, multi-criteria)
- [ ] 9.4 Add parameter scaling and normalization helpers

## 10. Testing and Examples
- [ ] 10.1 Test optimization: tune RC filter for target cutoff frequency
- [ ] 10.2 Test optimization: tune opamp compensation for PM target
- [ ] 10.3 Test sensitivity analysis with known analytical derivatives
- [ ] 10.4 Test DOE with visualization of design space coverage
- [ ] 10.5 Create example: "Optimize ADC comparator for ENOB"
- [ ] 10.6 Create example: "Parameter sensitivity ranking for amplifier"
- [ ] 10.7 Create example: "Multi-objective optimization (gain vs bandwidth)"

## 11. Documentation
- [ ] 11.1 Write `docs/optimization.md` with API and examples
- [ ] 11.2 Write `docs/sensitivity.md` explaining finite difference methods
- [ ] 11.3 Write `docs/doe.md` with DOE theory and usage
- [ ] 11.4 Document optimizer selection guide (gradient-free vs gradient-based)
- [ ] 11.5 Document multi-objective optimization workflows
