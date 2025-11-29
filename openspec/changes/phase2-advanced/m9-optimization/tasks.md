# M9: Optimization & Design of Experiments - Tasks

**Status:** In Progress (Phase 6 Complete)
**Start Date:** 2025-11-29
**Target Completion:** TBD (8-10 weeks)

## Task Breakdown

### Phase 1: Optimization Framework (Week 1-2) ✅ COMPLETE
- [x] Create `spicelab/optimization/` package
- [x] Implement base protocols (Optimizer, ObjectiveFunction)
- [x] Create OptimizationResult data model
- [x] Build CircuitOptimizer base class
- [x] Add parameter update mechanism
- [x] Write framework tests

**Completed:** 2025-11-29

Files created:
- `spicelab/optimization/__init__.py` - Package exports
- `spicelab/optimization/base.py` - Core classes (ParameterBounds, OptimizationResult, CircuitOptimizer, CircuitObjective)
- `spicelab/optimization/scipy_optimizers.py` - Scipy optimizer wrappers (Nelder-Mead, Powell, L-BFGS-B, DE, Dual Annealing)
- `tests/test_optimization.py` - 29 unit tests (17 pass, 12 skipped without scipy)

**Estimated Time:** 2 weeks

---

### Phase 2: Scipy Integration (Week 3-4) ✅ COMPLETE
- [x] Implement minimize() wrapper (Nelder-Mead, BFGS, etc.) - Done in Phase 1
- [x] Add differential_evolution() support - Done in Phase 1
- [x] Create constraint handling (inequality, equality, bounds)
- [x] Implement objective function caching
- [x] Add convergence tracking (ConvergenceData, ConvergenceTracker)
- [x] Multi-start optimization (LHS, grid, random sampling)
- [x] Write scipy optimizer tests
- [ ] Create example: opamp gain optimization (deferred)

**Completed:** 2025-11-29

Files created/updated:
- `spicelab/optimization/utils.py` - Utilities (MultiStartOptimizer, ConvergenceTracker, constraint builders, sensitivity analysis)
- `tests/test_optimization.py` - 49 tests total (30 pass, 19 skipped without scipy)

**Estimated Time:** 2 weeks

---

### Phase 3: Genetic Algorithms (Week 5-6) ✅ COMPLETE
- [x] Implement single-objective GA (GeneticOptimizer)
- [x] Add NSGA-II multi-objective (NSGA2Optimizer)
- [x] Create custom operators (SBX crossover, polynomial mutation)
- [x] Implement Pareto front with knee point detection
- [x] Write GA tests
- [ ] Pareto front visualization (deferred to Phase 7)
- [ ] Create example: multi-objective amplifier design (deferred)

**Completed:** 2025-11-29

Files created:
- `spicelab/optimization/genetic.py` - GA and NSGA-II optimizers (ParetoFront, MultiObjectiveResult, GeneticOptimizer, NSGA2Optimizer)
- `tests/test_optimization.py` - 61 tests total (42 pass, 19 skipped without scipy)

**Estimated Time:** 2 weeks

---

### Phase 4: Design of Experiments (Week 7) ✅ COMPLETE
- [x] Implement factorial designs (full_factorial, fractional_factorial)
- [x] Add Latin Hypercube Sampling (maximin, correlation criteria)
- [x] Implement Sobol sequences (with scipy fallback)
- [x] Create Central Composite Design (CCD)
- [x] Add Box-Behnken design
- [x] Add response surface modeling (linear, quadratic, interaction)
- [x] Write DoE tests
- [ ] Create Taguchi methods (deferred)
- [ ] Create example: filter optimization with DoE (deferred)

**Completed:** 2025-11-29

Files created:
- `spicelab/optimization/doe.py` - DoE methods (factorial, LHS, Sobol, CCD, Box-Behnken, RSM)
- `tests/test_optimization.py` - 82 tests total (63 pass, 19 skipped without scipy)

**Estimated Time:** 1 week

---

### Phase 5: Sensitivity Analysis (Week 8) ✅ COMPLETE
- [x] Implement Morris (one-at-a-time) method
- [x] Add Sobol variance-based method (first-order S1 & total-order ST)
- [x] Implement local gradient-based sensitivity
- [x] Add One-at-a-Time (OAT) sweep analysis
- [x] Visualization helpers (tornado data generation, text reports)
- [x] Write sensitivity tests
- [ ] Integrate SALib (deferred - native implementation sufficient)
- [ ] Create FAST method (deferred)
- [ ] Implement delta moment-independent indices (deferred)
- [ ] Create example: power supply sensitivity (deferred)

**Completed:** 2025-11-29

Files created:
- `spicelab/optimization/sensitivity.py` - Sensitivity analysis methods (MorrisResult, SobolResult, LocalSensitivity, OATResult, morris_analysis, sobol_analysis, local_sensitivity, oat_analysis, generate_tornado_data, print_sensitivity_report)
- `tests/test_optimization.py` - 86 tests total (67 pass, 19 skipped without scipy)

**Estimated Time:** 1 week

---

### Phase 6: Corner Analysis & PVT (Week 9) ✅ COMPLETE
- [x] Implement corner generation (all combinations)
- [x] Add PVT sweep framework (process/voltage/temperature)
- [x] Create statistical corner analysis (Monte Carlo)
- [x] Implement worst-case analysis (extremes & statistical methods)
- [x] Add yield estimation (get_yield, get_cpk)
- [x] Write corner analysis tests
- [ ] Create example: LDO corner analysis (deferred)

**Completed:** 2025-11-29

Files created:
- `spicelab/optimization/corner.py` - Corner analysis classes (Corner, CornerDefinition, CornerAnalysis, PVTSweep, PVTCondition, StatisticalCornerResult, WorstCaseResult, statistical_corner_analysis, worst_case_analysis, print_corner_report)
- `tests/test_optimization.py` - 119 tests total (100 pass, 19 skipped without scipy)

**Estimated Time:** 1 week

---

### Phase 7: Documentation & Integration (Week 10)
- [ ] Document all optimization methods
- [ ] Create optimization cookbook
- [ ] Add performance tuning guide
- [ ] Write integration tests
- [ ] Create benchmarks
- [ ] Update main documentation

**Estimated Time:** 1 week

---

## Acceptance Criteria

### Must Have
- [ ] Scipy.optimize integration (5+ methods)
- [ ] Genetic algorithms (single + multi-objective)
- [ ] DoE methods: factorial, LHS, Sobol
- [ ] Sensitivity analysis: Morris, Sobol
- [ ] Corner analysis automated
- [ ] PVT sweep helpers
- [ ] Test coverage ≥95%
- [ ] Typical optimization <1hr

### Should Have
- [ ] Response surface modeling
- [ ] Yield analysis
- [ ] Parallel optimization (multi-core)
- [ ] Optimization result caching

### Nice to Have
- [ ] Bayesian optimization
- [ ] Surrogate modeling
- [ ] Active learning

## Dependencies

- M7 (measurements)
- M2 (performance)
- scipy
- DEAP
- SALib
- pyDOE2

---

**Last Updated:** 2025-11-29 (Phase 6 Complete)
