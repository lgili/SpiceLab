# CLI Tools Specification

## ADDED Requirements

### Requirement: SpiceLab Doctor Diagnostic Tool
The system SHALL provide a `spicelab doctor` command to diagnose installation and environment.

#### Scenario: Check all engines
- **GIVEN** system with some engines installed
- **WHEN** user runs `spicelab doctor`
- **THEN** system SHALL check ngspice, LTspice, Xyce, libngspice, ADMS and report status for each

#### Scenario: Provide installation suggestions
- **GIVEN** ngspice not found in PATH
- **WHEN** doctor detects missing ngspice
- **THEN** system SHALL output: "ngspice: NOT FOUND. Install: [sudo apt install ngspice] (Linux), [brew install ngspice] (macOS), [download from...] (Windows)"

#### Scenario: Version compatibility check
- **GIVEN** ngspice version 35 installed
- **WHEN** doctor checks versions
- **THEN** system SHALL report version and compatibility: "ngspice 35 ✓ (supported: 34+)"

#### Scenario: Generate diagnostic report
- **GIVEN** doctor scan complete
- **WHEN** user requests JSON output: `spicelab doctor --json > report.json`
- **THEN** system SHALL output structured diagnostic data for bug reports

### Requirement: Format Conversion CLI
The system SHALL provide `spicelab convert` command for file format conversion.

#### Scenario: Convert RAW to Parquet
- **GIVEN** LTspice .raw file
- **WHEN** user runs `spicelab convert input.raw output.parquet`
- **THEN** system SHALL read RAW, convert to xarray, and save as Parquet with metadata

#### Scenario: Convert RAW to CSV
- **GIVEN** ngspice .raw file
- **WHEN** user runs `spicelab convert input.raw output.csv --format csv`
- **THEN** system SHALL export signals as CSV with time/freq in first column

#### Scenario: Batch conversion
- **GIVEN** directory with 100 .raw files
- **WHEN** user runs `spicelab convert *.raw --output-dir parquet/ --format parquet`
- **THEN** system SHALL convert all files in parallel with progress bar

### Requirement: Error Messages with Actionable Guidance
The system SHALL provide clear, actionable error messages with suggestions.

#### Scenario: Missing dependency error
- **GIVEN** user attempts operation requiring Plotly but not installed
- **WHEN** ImportError occurs
- **THEN** error message SHALL state: "Plotly required for this operation. Install: pip install spicelab[viz]"

#### Scenario: Typo suggestion
- **GIVEN** user references signal "V(ouut)" but correct is "V(out)"
- **WHEN** signal not found
- **THEN** error SHALL suggest: "Signal 'V(ouut)' not found. Did you mean: 'V(out)'?"

#### Scenario: Configuration issue hint
- **GIVEN** simulation fails due to convergence
- **WHEN** engine reports convergence failure
- **THEN** error SHALL include: "Convergence failed. Try: 1) Tighter tolerances, 2) Better initial conditions, 3) Smaller time step"
