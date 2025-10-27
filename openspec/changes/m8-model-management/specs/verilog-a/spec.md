# Verilog-A Support Specification

## ADDED Requirements

### Requirement: Verilog-A Model Compilation
The system SHALL support compilation of Verilog-A models using ADMS toolchain for compatible engines.

#### Scenario: Detect ADMS toolchain
- **GIVEN** ADMS installed in system PATH
- **WHEN** checking Verilog-A support
- **THEN** system SHALL detect `admsXml` binary and report version

#### Scenario: Missing ADMS toolchain
- **GIVEN** ADMS not installed
- **WHEN** attempting to use Verilog-A model
- **THEN** system SHALL raise error: "ADMS toolchain required for Verilog-A. Install: [instructions for OS]"

#### Scenario: Compile Verilog-A model for Xyce
- **GIVEN** Verilog-A source file `varactor.va`
- **WHEN** using model with Xyce engine
- **THEN** system SHALL invoke ADMS to compile model to Xyce-compatible format

#### Scenario: Cache compiled Verilog-A model
- **GIVEN** compiled Verilog-A model from source hash `abc123`
- **WHEN** using same source again
- **THEN** system SHALL reuse cached compiled model without recompilation

### Requirement: Verilog-A Error Handling
The system SHALL parse and present Verilog-A compilation errors in user-friendly format.

#### Scenario: Parse ADMS compilation error
- **GIVEN** Verilog-A file with syntax error at line 42
- **WHEN** compilation fails
- **THEN** system SHALL extract line number and error message: "varactor.va:42: syntax error near 'endmodule'"

#### Scenario: Detect missing discipline
- **GIVEN** Verilog-A module using undeclared discipline
- **WHEN** compilation reports missing discipline
- **THEN** system SHALL suggest: "Discipline 'thermal' not found. Add discipline include or check spelling"

#### Scenario: Report unsupported Verilog-A features
- **GIVEN** Verilog-A code using advanced features not supported by engine
- **WHEN** compilation or simulation fails
- **THEN** system SHALL identify unsupported feature: "Noise sources not supported in Xyce version X.Y"

### Requirement: Engine Verilog-A Compatibility
The system SHALL check and report Verilog-A support per engine with feature detection.

#### Scenario: Xyce supports Verilog-A via ADMS
- **GIVEN** Xyce engine with ADMS support
- **WHEN** checking Verilog-A capability
- **THEN** system SHALL report `supports_verilog_a=True`

#### Scenario: ngspice limited Verilog-A support
- **GIVEN** ngspice engine (standard build without Verilog-A)
- **WHEN** checking Verilog-A capability
- **THEN** system SHALL report `supports_verilog_a=False` and suggest Xyce alternative

#### Scenario: LTspice no Verilog-A support
- **GIVEN** LTspice engine
- **WHEN** attempting to use Verilog-A model
- **THEN** system SHALL raise error: "LTspice does not support Verilog-A. Consider: 1) Use Xyce engine, 2) Convert to .subckt equivalent"

### Requirement: Verilog-A Model Parameters
The system SHALL extract and validate parameters from Verilog-A module definitions.

#### Scenario: Extract parameters from Verilog-A source
- **GIVEN** Verilog-A module with parameters `R0=1k`, `TC=0.001`
- **WHEN** parsing module
- **THEN** system SHALL extract parameter names, defaults, and units

#### Scenario: Override Verilog-A parameter
- **GIVEN** Verilog-A model with parameter `R0=1k`
- **WHEN** user specifies `ModelRef(..., params={"R0": 2200})`
- **THEN** system SHALL pass parameter override to compiled model

#### Scenario: Validate parameter types
- **GIVEN** Verilog-A parameter defined as `integer`
- **WHEN** user provides float value
- **THEN** system SHALL warn: "Parameter 'n' expects integer, got float. Value will be truncated"

### Requirement: Verilog-A Model Introspection
The system SHALL provide introspection capabilities for Verilog-A modules.

#### Scenario: List module terminals
- **GIVEN** compiled Verilog-A model
- **WHEN** user requests terminal list
- **THEN** system SHALL return list of terminal names with disciplines: `[("p", "electrical"), ("n", "electrical")]`

#### Scenario: Extract module documentation
- **GIVEN** Verilog-A file with comment header describing module
- **WHEN** user requests model documentation
- **THEN** system SHALL extract and display comment block as documentation

### Requirement: Verilog-A Cross-Platform Support
The system SHALL handle platform-specific differences in Verilog-A toolchain.

#### Scenario: Linux ADMS installation
- **GIVEN** Linux system
- **WHEN** ADMS not found
- **THEN** error message SHALL suggest: "Install ADMS: sudo apt install adms (Debian/Ubuntu) or equivalent"

#### Scenario: macOS ADMS installation
- **GIVEN** macOS system
- **WHEN** ADMS not found
- **THEN** error message SHALL suggest: "Install ADMS: brew install adms"

#### Scenario: Windows ADMS installation
- **GIVEN** Windows system
- **WHEN** ADMS not found
- **THEN** error message SHALL provide download link and installation instructions
