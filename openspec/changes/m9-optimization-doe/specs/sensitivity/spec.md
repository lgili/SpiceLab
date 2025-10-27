# Sensitivity Analysis Specification

## ADDED Requirements

### Requirement: Local Sensitivity by Finite Differences
The system SHALL compute parameter sensitivity using finite difference approximations.

#### Scenario: Forward difference sensitivity
- **GIVEN** circuit with parameter `R1=1000` and measurement `PM`
- **WHEN** user requests sensitivity with `method="forward"`, `delta=0.01`
- **THEN** system SHALL compute `dPM/dR1 ≈ (PM(R1+δ) - PM(R1)) / δ`

#### Scenario: Central difference sensitivity (higher accuracy)
- **GIVEN** smooth objective function
- **WHEN** user requests sensitivity with `method="central"`
- **THEN** system SHALL compute `dPM/dR1 ≈ (PM(R1+δ) - PM(R1-δ)) / (2δ)` for better accuracy

#### Scenario: Multi-parameter Jacobian matrix
- **GIVEN** circuit with parameters `[R1, C1, R2]` and measurements `[PM, GBW]`
- **WHEN** user requests full sensitivity matrix
- **THEN** system SHALL return 2×3 Jacobian: `[[dPM/dR1, dPM/dC1, dPM/dR2], [dGBW/dR1, dGBW/dC1, dGBW/dR2]]`

### Requirement: Normalized Sensitivity
The system SHALL provide normalized sensitivity for comparing parameters with different scales.

#### Scenario: Percentage sensitivity
- **GIVEN** sensitivity `dPM/dR1 = 0.01 deg/Ω`
- **WHEN** user requests normalized sensitivity
- **THEN** system SHALL return `(dPM/dR1) × (R1/PM)` for scale-independent comparison

#### Scenario: Rank parameters by sensitivity
- **GIVEN** multiple parameter sensitivities
- **WHEN** user requests ranking
- **THEN** system SHALL sort parameters by absolute normalized sensitivity descending

### Requirement: Sensitivity Step Size Selection
The system SHALL provide guidance and automatic selection of finite difference step size.

#### Scenario: Automatic step size from parameter value
- **GIVEN** parameter `R1=1000`
- **WHEN** user doesn't specify step size
- **THEN** system SHALL use default relative step: `delta = 0.01 × R1 = 10`

#### Scenario: Adaptive step size for numerical stability
- **GIVEN** highly nonlinear measurement response
- **WHEN** fixed step size causes numerical errors
- **THEN** system SHALL adaptively reduce step size and warn if accuracy is compromised

### Requirement: Xyce Sensitivity Analysis Integration
The system SHALL support Xyce built-in `.sens` analysis for direct sensitivity computation.

#### Scenario: Run Xyce .sens analysis
- **GIVEN** Xyce engine and circuit with parameters
- **WHEN** user requests sensitivity via Xyce
- **THEN** system SHALL generate netlist with `.sens` directive and parse output

#### Scenario: Parse Xyce sensitivity output
- **GIVEN** Xyce `.sens.prn` output file
- **WHEN** reading sensitivity results
- **THEN** system SHALL parse and map to xarray.Dataset with dimensions `[measurement, parameter]`

#### Scenario: Fall back to finite differences for non-Xyce engines
- **GIVEN** ngspice or LTspice engine (no built-in .sens)
- **WHEN** user requests sensitivity
- **THEN** system SHALL automatically use finite difference method

### Requirement: Sensitivity Caching
The system SHALL cache simulation results for efficient sensitivity computation.

#### Scenario: Reuse base simulation for multiple parameters
- **GIVEN** sensitivity analysis for 3 parameters
- **WHEN** computing forward differences
- **THEN** system SHALL run base simulation once and reuse for all `f(x+δ)` comparisons

#### Scenario: Parallel sensitivity evaluation
- **GIVEN** 10 parameters requiring forward differences
- **WHEN** computing sensitivities
- **THEN** system SHALL simulate all 10 perturbed cases in parallel

### Requirement: Sensitivity Visualization
The system SHALL provide visualization tools for sensitivity results.

#### Scenario: Sensitivity bar chart
- **GIVEN** sensitivity results for multiple parameters
- **WHEN** user requests visualization
- **THEN** system SHALL generate bar chart showing normalized sensitivity magnitude per parameter

#### Scenario: Sensitivity heatmap for multi-output
- **GIVEN** Jacobian matrix (multiple measurements × parameters)
- **WHEN** user requests heatmap
- **THEN** system SHALL display color-coded heatmap showing sensitivity patterns

### Requirement: Sensitivity Error Handling
The system SHALL detect and report issues in sensitivity computation.

#### Scenario: Detect zero or near-zero derivatives
- **GIVEN** parameter with no influence on measurement
- **WHEN** computing sensitivity
- **THEN** system SHALL report `dPM/dR1 ≈ 0` and warn: "Parameter R1 has negligible effect on PM"

#### Scenario: Detect numerical instability
- **GIVEN** step size too small causing floating-point errors
- **WHEN** finite difference produces unreliable result
- **THEN** system SHALL warn: "Numerical instability detected. Consider larger step size or central difference"
