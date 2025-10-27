# Model Management Specification

## ADDED Requirements

### Requirement: Model Reference Type
The system SHALL provide a typed `ModelRef` class for referencing SPICE models, subcircuits, and Verilog-A modules.

#### Scenario: Reference SPICE model
- **GIVEN** a SPICE model file `models/diodes.lib` containing model `1N4148`
- **WHEN** user creates `ModelRef(kind="model", path="models/diodes.lib", name="1N4148")`
- **THEN** system SHALL validate the reference and store metadata

#### Scenario: Reference subcircuit with parameters
- **GIVEN** subcircuit `opamp741` with parameters `Aol`, `GBW`
- **WHEN** user creates `ModelRef(kind="subckt", path="opamps.lib", name="opamp741", params={"Aol": 1e5})`
- **THEN** system SHALL validate parameters against subcircuit definition

#### Scenario: Reference Verilog-A module
- **GIVEN** Verilog-A file `resistor_thermal.va`
- **WHEN** user creates `ModelRef(kind="veriloga", path="resistor_thermal.va", name="res_thermal")`
- **THEN** system SHALL validate file exists and name matches module name

### Requirement: Model Registry
The system SHALL provide a centralized model registry with search path resolution and versioning.

#### Scenario: Register model library
- **GIVEN** model library at `/usr/share/spicelab/models/standard.lib`
- **WHEN** user adds library to registry
- **THEN** registry SHALL index all models and subcircuits in the file

#### Scenario: Resolve model by name
- **GIVEN** registered libraries in search paths: `[./models, ~/.spicelab/models, /usr/share/spicelab/models]`
- **WHEN** user requests model `opamp741`
- **THEN** registry SHALL search paths in order and return first match with full path

#### Scenario: Handle model name collision
- **GIVEN** model `opamp741` exists in both local and system libraries
- **WHEN** user requests `opamp741` without specifying path
- **THEN** registry SHALL return local version and warn about collision

#### Scenario: Validate model version compatibility
- **GIVEN** model library with version metadata
- **WHEN** loading model with version constraint `>=2.0`
- **THEN** registry SHALL check version and reject if incompatible

### Requirement: Model Search Path Configuration
The system SHALL support configurable model search paths with precedence rules.

#### Scenario: Default search path order
- **GIVEN** no user configuration
- **WHEN** searching for models
- **THEN** system SHALL search in order: `./models`, `~/.spicelab/models`, system install directory

#### Scenario: Add custom search path
- **GIVEN** user has models in `/project/pdk/models`
- **WHEN** user adds path to configuration
- **THEN** system SHALL prepend to search path with highest priority

#### Scenario: Environment variable override
- **GIVEN** environment variable `SPICELAB_MODEL_PATH=/opt/models:/home/user/lib`
- **WHEN** initializing registry
- **THEN** system SHALL parse paths and insert into search order

### Requirement: Model File Inclusion in Netlists
The system SHALL generate appropriate `.include` and `.lib` directives for each engine.

#### Scenario: Include model library in ngspice netlist
- **GIVEN** Circuit with `ModelRef(path="models/opamps.lib")`
- **WHEN** generating ngspice netlist
- **THEN** netlist SHALL contain `.include models/opamps.lib` before component usage

#### Scenario: Include library in LTspice netlist
- **GIVEN** Circuit with `ModelRef(kind="model", path="diodes.lib", name="1N4148")`
- **WHEN** generating LTspice netlist
- **THEN** netlist SHALL contain `.lib diodes.lib` with LTspice-compatible syntax

#### Scenario: Order includes by dependency
- **GIVEN** multiple ModelRef instances with dependencies
- **WHEN** generating netlist
- **THEN** system SHALL order `.include` directives so dependencies appear first

### Requirement: Model Validation
The system SHALL validate models before simulation to catch errors early.

#### Scenario: Detect missing model file
- **GIVEN** `ModelRef(path="nonexistent.lib")`
- **WHEN** validating circuit
- **THEN** system SHALL raise error: "Model file not found: nonexistent.lib. Search paths: [...]"

#### Scenario: Detect model name mismatch
- **GIVEN** `ModelRef(path="opamps.lib", name="opamp_xyz")`
- **WHEN** validating and `opamp_xyz` not in file
- **THEN** system SHALL raise error: "Model 'opamp_xyz' not found in opamps.lib. Available: ['opamp741', 'lm324']"

#### Scenario: Validate parameter names
- **GIVEN** subcircuit with parameters `[Aol, GBW]`
- **WHEN** user provides `params={"gain": 100}` (typo)
- **THEN** system SHALL warn: "Unknown parameter 'gain'. Expected: Aol, GBW"

### Requirement: Model Caching
The system SHALL cache model file contents and compiled artifacts to improve performance.

#### Scenario: Cache parsed model library
- **GIVEN** large model library file (10 MB)
- **WHEN** loading library for first time
- **THEN** system SHALL parse and cache index by file hash

#### Scenario: Reuse cached model on unchanged file
- **GIVEN** cached model library
- **WHEN** loading same library with matching hash
- **THEN** system SHALL skip parsing and use cached index

#### Scenario: Invalidate cache on file modification
- **GIVEN** cached model library
- **WHEN** library file is modified (hash changes)
- **THEN** system SHALL detect change, invalidate cache, and re-parse
