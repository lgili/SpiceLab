# Yield & Design for Manufacturing Specification

## ADDED Requirements

### Requirement: Yield Estimation from Monte Carlo
The system SHALL estimate manufacturing yield from Monte Carlo simulation with component tolerances.

#### Scenario: Monte Carlo with tolerance specs
- **GIVEN** resistors with ±5% tolerance, capacitors with ±10%
- **WHEN** running 10,000 Monte Carlo trials
- **THEN** system SHALL generate random values from normal distributions matching tolerances

#### Scenario: Compute yield vs specifications
- **GIVEN** spec limits: gain > 60 dB, PM > 45°
- **WHEN** analyzing 10,000 MC results
- **THEN** system SHALL report yield = (# passing trials) / (total trials) × 100%

#### Scenario: Identify yield limiters
- **GIVEN** yield of 85%
- **WHEN** analyzing failure modes
- **THEN** system SHALL rank parameters by contribution to failures: "R5 tolerance causes 60% of PM failures"

### Requirement: Worst-Case Analysis
The system SHALL perform worst-case analysis combining corners and tolerances.

#### Scenario: Worst-case corners with component tolerances
- **GIVEN** circuit in SS corner with all resistors at -5% tolerance
- **WHEN** running worst-case analysis
- **THEN** system SHALL test all extreme combinations and report min/max performance

#### Scenario: Pareto of failure contributors
- **GIVEN** circuit failing specs in worst case
- **WHEN** analyzing sensitivity
- **THEN** system SHALL generate Pareto chart showing which parameters contribute most to failure

### Requirement: Robust Design Optimization
The system SHALL support design centering and robust optimization.

#### Scenario: Taguchi robust design
- **GIVEN** circuit parameters as control factors
- **WHEN** running Taguchi DOE
- **THEN** system SHALL find parameter values minimizing sensitivity to noise factors

#### Scenario: Design centering
- **GIVEN** initial design at edge of spec margin
- **WHEN** running design centering
- **THEN** system SHALL shift nominal parameter values to maximize margin from spec limits

#### Scenario: Multi-objective yield optimization
- **GIVEN** objectives: maximize yield, minimize cost
- **WHEN** optimizing
- **THEN** system SHALL find Pareto front trading off yield vs component cost

### Requirement: Experiment Tracking and Reproducibility
The system SHALL track design iterations with full reproducibility.

#### Scenario: Log design iteration
- **GIVEN** optimization run with parameters, measurements, yield
- **WHEN** logging iteration
- **THEN** system SHALL store: parameters, netlist hash, measurements, yield, timestamp

#### Scenario: Compare design iterations
- **GIVEN** 10 logged design iterations
- **WHEN** viewing comparison
- **THEN** system SHALL show table/plot of parameter evolution, yield trend, Pareto improvements

#### Scenario: Reproduce design from log
- **GIVEN** logged iteration ID
- **WHEN** requesting reproduction
- **THEN** system SHALL retrieve exact netlist, parameters, seed, and re-run to verify identical results
