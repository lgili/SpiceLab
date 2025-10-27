# Design of Experiments (DOE) Specification

## ADDED Requirements

### Requirement: Full Factorial Design
The system SHALL support full factorial experimental designs for exhaustive parameter space exploration.

#### Scenario: Two-level full factorial
- **GIVEN** parameters `R1=[1k, 10k]`, `C1=[10p, 100p]`
- **WHEN** user creates full factorial DOE
- **THEN** system SHALL generate 2² = 4 combinations: `[(1k,10p), (1k,100p), (10k,10p), (10k,100p)]`

#### Scenario: Multi-level full factorial
- **GIVEN** parameter `R1=[1k, 5k, 10k]`, `C1=[10p, 50p, 100p]`
- **WHEN** user creates full factorial DOE
- **THEN** system SHALL generate 3² = 9 combinations covering all level combinations

#### Scenario: Combinatorial explosion warning
- **GIVEN** 5 parameters with 10 levels each
- **WHEN** full factorial would generate 10⁵ = 100,000 runs
- **THEN** system SHALL warn: "Full factorial requires 100,000 runs. Consider LHS or Sobol sampling for large designs"

### Requirement: Latin Hypercube Sampling (LHS)
The system SHALL support Latin Hypercube Sampling for efficient space-filling designs.

#### Scenario: Generate LHS design
- **GIVEN** parameters `R1=[100, 10k]`, `C1=[1p, 1n]` and `n_samples=50`
- **WHEN** user creates LHS DOE
- **THEN** system SHALL generate 50 samples with stratified coverage using `scipy.stats.qmc.LatinHypercube`

#### Scenario: LHS with reproducible seed
- **GIVEN** LHS design with `seed=42`
- **WHEN** generating samples
- **THEN** system SHALL produce identical sample set on repeated runs with same seed

#### Scenario: LHS optimization for space-filling
- **GIVEN** LHS with `optimize=True`
- **WHEN** generating samples
- **THEN** system SHALL apply criterion optimization (e.g., maximin) to improve uniformity

### Requirement: Sobol Sequence Sampling
The system SHALL support low-discrepancy Sobol sequences for quasi-random designs.

#### Scenario: Generate Sobol sequence
- **GIVEN** 3 parameters and `n_samples=128`
- **WHEN** user creates Sobol DOE
- **THEN** system SHALL generate low-discrepancy sequence using `scipy.stats.qmc.Sobol`

#### Scenario: Sobol sequence power-of-two sizes
- **GIVEN** user requests 100 samples
- **WHEN** Sobol works best with powers of 2
- **THEN** system SHALL suggest: "Sobol sequence recommended size: 128 (next power of 2). Proceed with 100? [y/n]"

### Requirement: DOE Integration with Orchestrator
The system SHALL integrate DOE designs with simulation orchestrator for execution.

#### Scenario: Execute DOE plan
- **GIVEN** LHS design with 50 parameter combinations
- **WHEN** user runs DOE
- **THEN** orchestrator SHALL simulate all 50 cases and collect results in structured dataset

#### Scenario: Parallel DOE execution
- **GIVEN** DOE with 100 samples
- **WHEN** executing on system with 8 cores
- **THEN** orchestrator SHALL run up to 8 simulations in parallel

### Requirement: DOE Result Analysis
The system SHALL provide analysis tools for DOE results.

#### Scenario: Compute main effects
- **GIVEN** DOE results with measurements
- **WHEN** user requests main effects analysis
- **THEN** system SHALL compute average effect of each parameter on output

#### Scenario: Compute interaction effects (two-way)
- **GIVEN** full factorial DOE results
- **WHEN** user requests interaction analysis
- **THEN** system SHALL compute two-way interaction effects between parameter pairs

#### Scenario: Response surface approximation
- **GIVEN** DOE results with input parameters and output measurements
- **WHEN** user fits response surface model
- **THEN** system SHALL fit polynomial or RBF model for prediction at untested points

### Requirement: DOE Visualization
The system SHALL provide visualization tools for DOE results and coverage.

#### Scenario: Scatter matrix plot
- **GIVEN** DOE with 3 parameters and results
- **WHEN** user requests scatter matrix
- **THEN** system SHALL plot all pairwise parameter combinations with color-coded output

#### Scenario: Parameter space coverage visualization
- **GIVEN** LHS or Sobol sample set
- **WHEN** user visualizes coverage
- **THEN** system SHALL show 2D projections highlighting uniform distribution

#### Scenario: Pareto front from DOE
- **GIVEN** multi-objective DOE results
- **WHEN** user requests Pareto visualization
- **THEN** system SHALL identify and plot non-dominated solutions

### Requirement: DOE Export and Reproducibility
The system SHALL support export and reproducibility of DOE designs and results.

#### Scenario: Export DOE plan to CSV
- **GIVEN** generated DOE plan
- **WHEN** user exports to CSV before simulation
- **THEN** system SHALL save parameter combinations with run IDs

#### Scenario: Export DOE results to Parquet
- **GIVEN** completed DOE with measurements
- **WHEN** user exports results
- **THEN** system SHALL save Parquet with columns: `[run_id, R1, C1, ..., PM, GBW, ...]`

#### Scenario: Reproducible DOE from seed
- **GIVEN** DOE specification with `seed=123`
- **WHEN** re-running DOE
- **THEN** system SHALL generate identical sample set and ordering

### Requirement: DOE Strategy Selection Guidance
The system SHALL provide guidance on selecting appropriate DOE strategy.

#### Scenario: Recommend strategy based on problem size
- **GIVEN** 2 parameters with moderate ranges
- **WHEN** user asks for DOE recommendation
- **THEN** system SHALL suggest: "Full factorial (feasible). For screening, consider LHS with 20-50 samples"

#### Scenario: Recommend strategy for many parameters
- **GIVEN** 10 parameters to explore
- **WHEN** user asks for DOE recommendation
- **THEN** system SHALL suggest: "LHS or Sobol sampling with n ≥ 10×d = 100 samples. Full factorial not feasible"

### Requirement: Adaptive DOE (Future)
The system SHALL support adaptive DOE strategies that refine sampling based on results.

#### Scenario: Identify region of interest
- **GIVEN** initial DOE results
- **WHEN** specific region shows interesting behavior
- **THEN** system SHALL suggest refined sampling in that region

#### Scenario: Sequential DOE refinement
- **GIVEN** coarse initial DOE
- **WHEN** user requests refinement
- **THEN** system SHALL generate additional samples in under-explored or high-gradient regions
