# Optimization Specification

## ADDED Requirements

### Requirement: Optimization Specification API
The system SHALL provide a typed API for defining circuit parameter optimization problems.

#### Scenario: Define single-objective optimization
- **GIVEN** circuit with tunable parameters `[R1, C1]`
- **WHEN** user creates `OptimizationSpec(params={"R1": (100, 10e3), "C1": (1e-12, 1e-6)}, objective=MinimizeError(...))`
- **THEN** system SHALL validate bounds and objective function

#### Scenario: Define multi-objective optimization
- **GIVEN** circuit to optimize for gain and bandwidth
- **WHEN** user specifies objectives `[MaximizeGain(), MaximizeBandwidth()]`
- **THEN** system SHALL configure multi-objective solver to find Pareto front

### Requirement: SciPy Optimizer Integration
The system SHALL provide wrappers for SciPy optimization algorithms with unified API.

#### Scenario: Local optimization with Nelder-Mead
- **GIVEN** optimization problem with smooth objective
- **WHEN** user selects `method="Nelder-Mead"`
- **THEN** system SHALL use `scipy.optimize.minimize` with Nelder-Mead and track iterations

#### Scenario: Bounded optimization with L-BFGS-B
- **GIVEN** parameters with box constraints
- **WHEN** user selects `method="L-BFGS-B"` with bounds
- **THEN** system SHALL enforce bounds and use gradient-based optimization

#### Scenario: Global optimization with Differential Evolution
- **GIVEN** multi-modal objective landscape
- **WHEN** user selects `method="differential_evolution"`
- **THEN** system SHALL use population-based global search within bounds

#### Scenario: Constrained optimization with SLSQP
- **GIVEN** optimization with equality/inequality constraints
- **WHEN** user provides constraints and selects `method="SLSQP"`
- **THEN** system SHALL enforce constraints during optimization

### Requirement: Objective Function Builder
The system SHALL provide tools to construct objective functions from simulation and measurements.

#### Scenario: Build objective from target measurement
- **GIVEN** target phase margin of 60 degrees
- **WHEN** user creates objective `MinimizeError(target={"PM": 60}, weight={"PM": 1.0})`
- **THEN** system SHALL generate objective that runs simulation, computes PM, and returns `|PM_actual - 60|`

#### Scenario: Multi-criteria objective with weights
- **GIVEN** multiple targets with different priorities
- **WHEN** user specifies `objective=MinimizeError(target={"PM": 60, "GBW": 1e6}, weight={"PM": 2.0, "GBW": 1.0})`
- **THEN** system SHALL compute weighted sum: `2.0 * |PM - 60| + 1.0 * |GBW - 1e6|`

#### Scenario: Custom objective function
- **GIVEN** user-defined objective logic
- **WHEN** user provides `objective=lambda params: custom_loss(params)`
- **THEN** system SHALL accept callable with signature `(params: dict) -> float`

### Requirement: Optimization Execution
The system SHALL execute optimization with automatic circuit instantiation and simulation.

#### Scenario: Run optimization loop
- **GIVEN** OptimizationSpec with bounds and objective
- **WHEN** user calls `optimize.run(spec, max_iter=100)`
- **THEN** system SHALL iteratively: 1) propose parameters, 2) build circuit, 3) simulate, 4) measure, 5) compute loss, until convergence

#### Scenario: Parallel objective evaluation
- **GIVEN** population-based algorithm (e.g., differential evolution)
- **WHEN** evaluating population of 20 parameter sets
- **THEN** system SHALL simulate all 20 in parallel using orchestrator

### Requirement: Optimization Result Tracking
The system SHALL track optimization history and provide detailed results.

#### Scenario: Record optimization history
- **GIVEN** optimization running for 50 iterations
- **WHEN** optimization completes
- **THEN** result SHALL include DataFrame with columns: `[iteration, R1, C1, PM, loss]` for all iterations

#### Scenario: Extract best parameters
- **GIVEN** completed optimization
- **WHEN** user requests best parameters
- **THEN** system SHALL return parameter dict with lowest loss: `{"R1": 4700, "C1": 100e-12}`

#### Scenario: Check convergence status
- **GIVEN** optimization terminated
- **WHEN** user checks result
- **THEN** result SHALL indicate convergence status: `converged=True`, `termination_reason="ftol reached"`

### Requirement: Optimization Caching
The system SHALL cache simulation results during optimization to avoid redundant computation.

#### Scenario: Cache simulation by parameter hash
- **GIVEN** parameter set `{R1: 1000, C1: 1e-9}`
- **WHEN** optimizer re-evaluates same parameters
- **THEN** system SHALL return cached measurement result without re-simulation

#### Scenario: Invalidate cache on circuit change
- **GIVEN** cached optimization results
- **WHEN** circuit netlist changes
- **THEN** system SHALL invalidate cache and re-run simulations

### Requirement: Multi-Objective Optimization
The system SHALL support multi-objective optimization with Pareto front computation.

#### Scenario: Optimize for gain and bandwidth trade-off
- **GIVEN** opamp circuit with conflicting objectives
- **WHEN** user specifies `objectives=[MaximizeGain(), MaximizeBandwidth()]`
- **THEN** system SHALL compute Pareto front of non-dominated solutions

#### Scenario: Visualize Pareto front
- **GIVEN** multi-objective optimization result
- **WHEN** user requests Pareto visualization
- **THEN** system SHALL plot 2D or 3D scatter of objective space with Pareto front highlighted

### Requirement: Optimization Restart and Robustness
The system SHALL provide mechanisms for handling stuck optimizations and local minima.

#### Scenario: Restart from multiple initial points
- **GIVEN** optimization with local minima risk
- **WHEN** user enables multi-start with 5 initial points
- **THEN** system SHALL run 5 optimizations from different starting points and return global best

#### Scenario: Early stopping on convergence
- **GIVEN** optimization with `ftol=1e-6` tolerance
- **WHEN** function value changes less than tolerance for 10 iterations
- **THEN** system SHALL stop and report convergence

#### Scenario: Handle simulation failures during optimization
- **GIVEN** parameter values causing simulation divergence
- **WHEN** simulation fails
- **THEN** system SHALL return penalty loss (e.g., `inf`) and continue optimization

### Requirement: Optimization Error Handling
The system SHALL provide clear diagnostics for optimization failures.

#### Scenario: Detect infeasible constraints
- **GIVEN** optimization with contradictory constraints
- **WHEN** no feasible solution exists
- **THEN** system SHALL raise error: "No feasible solution found. Constraints may be contradictory"

#### Scenario: Warn about slow convergence
- **GIVEN** optimization making minimal progress
- **WHEN** 100 iterations elapsed with < 1% improvement
- **THEN** system SHALL warn: "Slow convergence detected. Consider: 1) Tighter bounds, 2) Different algorithm, 3) Scaling parameters"
