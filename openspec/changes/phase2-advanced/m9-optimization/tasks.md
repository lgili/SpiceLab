# M9: Optimization & Design of Experiments - Tasks

**Status:** In Progress (Phase 2 Complete)
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

### Phase 3: Genetic Algorithms (Week 5-6)
- [ ] Integrate DEAP library
- [ ] Implement single-objective GA
- [ ] Add NSGA-II (multi-objective)
- [ ] Create custom operators (mutation, crossover)
- [ ] Pareto front visualization
- [ ] Write GA tests
- [ ] Create example: multi-objective amplifier design

**Estimated Time:** 2 weeks

---

### Phase 4: Design of Experiments (Week 7)
- [ ] Implement factorial designs (2^k, 3^k)
- [ ] Add Latin Hypercube Sampling (LHS)
- [ ] Implement Sobol sequences
- [ ] Create Taguchi methods (orthogonal arrays)
- [ ] Add response surface modeling
- [ ] Write DoE tests
- [ ] Create example: filter optimization with DoE

**Estimated Time:** 1 week

---

### Phase 5: Sensitivity Analysis (Week 8)
- [ ] Integrate SALib
- [ ] Implement Morris (one-at-a-time) method
- [ ] Add Sobol variance-based method
- [ ] Create FAST (Fourier Amplitude Sensitivity Test)
- [ ] Implement delta moment-independent indices
- [ ] Visualization (tornado plots, scatter plots)
- [ ] Write sensitivity tests
- [ ] Create example: power supply sensitivity

**Estimated Time:** 1 week

---

### Phase 6: Corner Analysis & PVT (Week 9)
- [ ] Implement corner generation (all combinations)
- [ ] Add PVT sweep framework
- [ ] Create statistical corner analysis
- [ ] Implement worst-case analysis
- [ ] Add yield estimation
- [ ] Write corner analysis tests
- [ ] Create example: LDO corner analysis

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

**Last Updated:** 2025-11-29
