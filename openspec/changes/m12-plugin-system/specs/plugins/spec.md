# Plugin System Specification

## ADDED Requirements

### Requirement: Plugin Discovery via Entry Points
The system SHALL discover plugins using Python entry points mechanism.

#### Scenario: Auto-discover measurement plugin
- **GIVEN** third-party package with entry point `spicelab.plugins.measurements = mypackage:MyMeasurement`
- **WHEN** SpiceLab imports plugins
- **THEN** system SHALL discover and register `MyMeasurement` automatically

#### Scenario: List all installed plugins
- **GIVEN** multiple plugins installed
- **WHEN** user runs `spicelab plugins list`
- **THEN** system SHALL display: plugin name, version, type, author, description

#### Scenario: Plugin version compatibility check
- **GIVEN** plugin requiring `spicelab>=2.0,<3.0`
- **WHEN** loading plugin with SpiceLab 1.5
- **THEN** system SHALL warn: "Plugin 'custom_meas' requires spicelab>=2.0, current version: 1.5. Plugin disabled"

### Requirement: Plugin Base Classes
The system SHALL provide base classes for each plugin type with clear interfaces.

#### Scenario: Create custom measurement plugin
- **GIVEN** developer wants custom measurement
- **WHEN** subclassing `MeasurementPlugin` and implementing `compute(ds: xr.Dataset) -> float`
- **THEN** plugin SHALL be compatible with measurement framework

#### Scenario: Create custom reader plugin
- **GIVEN** proprietary file format to support
- **WHEN** subclassing `ReaderPlugin` and implementing `read(path: str) -> xr.Dataset`
- **THEN** plugin SHALL integrate with I/O layer

#### Scenario: Create custom engine plugin
- **GIVEN** alternative SPICE simulator
- **WHEN** subclassing `EnginePlugin` and implementing `run(circuit, analysis) -> Result`
- **THEN** plugin SHALL work with orchestrator

### Requirement: Plugin Registration and Validation
The system SHALL validate plugins before registration.

#### Scenario: Validate required methods
- **GIVEN** measurement plugin missing `compute()` method
- **WHEN** attempting to register
- **THEN** system SHALL raise error: "Plugin 'MyMeas' must implement compute(ds: xr.Dataset) -> float"

#### Scenario: Detect plugin name conflicts
- **GIVEN** two plugins with name "phase_margin"
- **WHEN** registering second plugin
- **THEN** system SHALL warn: "Plugin 'phase_margin' already registered by package 'foo'. Skipping 'bar.phase_margin'"

#### Scenario: Plugin dependency checking
- **GIVEN** plugin requiring NumPy, SciPy
- **WHEN** loading plugin
- **THEN** system SHALL check dependencies and report missing ones

### Requirement: Plugin Error Isolation
The system SHALL isolate plugin errors to prevent system crashes.

#### Scenario: Plugin load failure
- **GIVEN** plugin with import error
- **WHEN** discovering plugins
- **THEN** system SHALL log error, skip plugin, continue loading others

#### Scenario: Plugin runtime error
- **GIVEN** measurement plugin raising exception
- **WHEN** executing measurement
- **THEN** system SHALL catch exception, return `None`, and log: "Plugin 'custom_meas' failed: [error message]"

### Requirement: API Stability and Deprecation
The system SHALL follow SemVer and provide clear deprecation warnings.

#### Scenario: Deprecate function with warning
- **GIVEN** function `old_api()` deprecated in v1.5, removed in v2.0
- **WHEN** user calls `old_api()` in v1.5
- **THEN** system SHALL emit `DeprecationWarning`: "old_api() deprecated in 1.5, will be removed in 2.0. Use new_api() instead"

#### Scenario: Provide compatibility shim
- **GIVEN** renamed function `analyze()` → `run_analysis()`
- **WHEN** user calls deprecated `analyze()`
- **THEN** shim in `_compat` SHALL forward to `run_analysis()` with warning

#### Scenario: Breaking change detection
- **GIVEN** PR modifying public API signature
- **WHEN** CI runs API stability tests
- **THEN** tests SHALL fail if breaking change without major version bump

### Requirement: Plugin Documentation and Discovery
The system SHALL provide tools to help users find and use plugins.

#### Scenario: Plugin help/documentation
- **GIVEN** installed plugin
- **WHEN** user runs `spicelab plugins info custom_meas`
- **THEN** system SHALL display: description, parameters, usage example, author, repo link

#### Scenario: Plugin marketplace/catalog
- **GIVEN** curated list of compatible plugins
- **WHEN** user searches for "noise measurement"
- **THEN** system SHALL suggest available plugins with install commands

### Requirement: Versioning and Release Process
The system SHALL follow automated release workflow with changelog.

#### Scenario: Automated release on tag
- **GIVEN** git tag `v2.1.0` pushed
- **WHEN** CI detects tag
- **THEN** system SHALL: build wheels, generate changelog from commits, publish to PyPI, create GitHub release

#### Scenario: Changelog generation
- **GIVEN** commits since last release with conventional commit format
- **WHEN** generating changelog
- **THEN** system SHALL categorize by: Features, Bug Fixes, Breaking Changes, Documentation

#### Scenario: Version compatibility matrix
- **GIVEN** multiple plugin versions
- **WHEN** user checks compatibility
- **THEN** docs SHALL provide matrix: SpiceLab version × plugin version compatibility
