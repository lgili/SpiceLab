# PDK & Model Library Management Specification

## ADDED Requirements

### Requirement: PDK Registry with Versioning
The system SHALL manage PDK libraries with version control and corner support.

#### Scenario: Register PDK library
- **GIVEN** foundry PDK with models for 65nm process
- **WHEN** registering PDK: `pdk.register("tsmc65nm", version="1.7", path="/pdk/tsmc65")`
- **THEN** system SHALL index models and corners

#### Scenario: Select process corner
- **GIVEN** PDK with TT (typical), SS (slow), FF (fast) corners
- **WHEN** user specifies `corner="SS"`
- **THEN** circuit SHALL use slow-slow corner models

#### Scenario: Corner matrix simulation
- **GIVEN** PDK corners [TT, SS, FF] × VDD [1.1V, 1.2V, 1.3V] × Temp [-40°C, 27°C, 125°C]
- **WHEN** running corner sweep
- **THEN** system SHALL simulate all 27 combinations

### Requirement: Parameter Extraction from Measurements
The system SHALL fit model parameters to experimental data.

#### Scenario: Extract diode model from IV curve
- **GIVEN** measured diode IV data points
- **WHEN** fitting SPICE diode model
- **THEN** system SHALL optimize parameters [Is, N, Rs] to minimize error vs measured data

#### Scenario: Extract MOS model from CV
- **GIVEN** measured gate capacitance vs voltage
- **WHEN** extracting MOS parameters
- **THEN** system SHALL fit oxide thickness, doping profile to match CV curve

#### Scenario: Report extraction quality
- **GIVEN** completed parameter extraction
- **WHEN** generating report
- **THEN** system SHALL provide R² goodness-of-fit, residual plots, parameter confidence intervals

### Requirement: Process Corner Presets
The system SHALL provide standard corner definitions for common analyses.

#### Scenario: Apply worst-case power corner
- **GIVEN** circuit to analyze for max power
- **WHEN** applying preset `corner="worst_power"`
- **THEN** system SHALL select FF corner, max VDD, max temp

#### Scenario: Apply worst-case speed corner
- **GIVEN** circuit to analyze for min speed
- **WHEN** applying preset `corner="worst_speed"`
- **THEN** system SHALL select SS corner, min VDD, max temp

#### Scenario: Custom corner definition
- **GIVEN** user-defined corner matrix
- **WHEN** saving as preset
- **THEN** system SHALL persist definition for reuse across projects
