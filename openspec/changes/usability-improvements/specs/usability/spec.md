# Usability Improvements Specification

## ADDED Requirements

### Requirement: Circuit Templates
The system SHALL provide pre-built circuit templates for common topologies.

#### Scenario: Create RC filter from template
- **GIVEN** user needs low-pass filter with fc=1kHz
- **WHEN** calling `Circuit.from_template("rc_lowpass", fc=1000)`
- **THEN** system SHALL generate RC circuit with calculated R and C values

#### Scenario: Create opamp amplifier from template
- **GIVEN** user needs non-inverting amplifier with gain=10
- **WHEN** calling `Circuit.from_template("opamp_noninv", gain=10, opamp="741")`
- **THEN** system SHALL generate circuit with feedback resistors calculated for gain

#### Scenario: List available templates
- **GIVEN** user exploring templates
- **WHEN** calling `Circuit.list_templates()`
- **THEN** system SHALL return list with names, descriptions, required parameters

### Requirement: Smart Unit Parsing
The system SHALL accept multiple unit formats and convert automatically.

#### Scenario: Accept various resistance formats
- **GIVEN** parameter specifications: `"1k"`, `"1kohm"`, `"1e3"`, `1000`
- **WHEN** parsing resistance value
- **THEN** all SHALL be interpreted as 1000 Ω

#### Scenario: Suggest correction for common typos
- **GIVEN** user enters `C1="10u"` (micro, not µ)
- **WHEN** parsing
- **THEN** system SHALL interpret as 10µF and optionally suggest: "Use '10µ' or '10e-6' for clarity"

#### Scenario: Engineering notation support
- **GIVEN** values like `"1.5M"`, `"22p"`, `"470n"`
- **WHEN** parsing
- **THEN** system SHALL correctly interpret as 1.5e6, 22e-12, 470e-9

### Requirement: Circuit Validation
The system SHALL validate circuit topology before simulation.

#### Scenario: Detect floating node
- **GIVEN** circuit with node connected to only one component terminal
- **WHEN** validating before simulation
- **THEN** system SHALL warn: "Node 'n5' is floating (connected to only one component). This may cause convergence issues"

#### Scenario: Detect voltage source short
- **GIVEN** two voltage sources with different values in parallel
- **WHEN** validating
- **THEN** system SHALL raise error: "Voltage sources V1 (5V) and V2 (3V) are in parallel. This creates a contradiction"

#### Scenario: Detect missing ground
- **GIVEN** circuit without ground (0) node
- **WHEN** validating
- **THEN** system SHALL warn: "No ground node (0) found. Add ground reference for DC operating point"

### Requirement: Workflow Shortcuts
The system SHALL provide shortcuts for common simulation workflows.

#### Scenario: Quick AC analysis
- **GIVEN** circuit to analyze
- **WHEN** calling `circuit.quick_ac()`
- **THEN** system SHALL run AC sweep with sensible defaults (1Hz-1GHz, decade points) and return plotted results

#### Scenario: Quick transient analysis
- **GIVEN** circuit with time-domain signals
- **WHEN** calling `circuit.quick_tran(duration="10ms")`
- **THEN** system SHALL run transient with auto-selected timestep and plot all node voltages

#### Scenario: Method chaining for workflows
- **GIVEN** circuit ready to simulate
- **WHEN** calling `result = circuit.simulate(AC(...)).measure(PhaseMargin()).plot()`
- **THEN** system SHALL execute in sequence: simulate → measure → plot and return result

### Requirement: Common Measurement Shortcuts
The system SHALL provide shortcut methods for frequently used measurements.

#### Scenario: Measure phase margin shortcut
- **GIVEN** AC analysis result with loop gain
- **WHEN** calling `result.pm(signal="loop_gain")`
- **THEN** system SHALL compute and return phase margin value

#### Scenario: Measure bandwidth shortcut
- **GIVEN** AC analysis result
- **WHEN** calling `result.bw(signal="V(out)", ref="V(in)", db=-3)`
- **THEN** system SHALL return -3dB bandwidth

#### Scenario: Measure overshoot shortcut
- **GIVEN** transient step response
- **WHEN** calling `result.overshoot(signal="V(out)")`
- **THEN** system SHALL compute and return overshoot percentage

### Requirement: Enhanced Error Messages
The system SHALL provide actionable, context-rich error messages.

#### Scenario: Missing dependency error
- **GIVEN** operation requiring optional dependency
- **WHEN** dependency not installed
- **THEN** error SHALL state: "Plotly required for 3D plots. Install with: pip install spicelab[viz]"

#### Scenario: Simulation convergence failure
- **GIVEN** simulation fails to converge
- **WHEN** error occurs
- **THEN** message SHALL include: "Convergence failed at t=1.23ms. Try: 1) Use .ic for initial conditions, 2) Reduce max timestep, 3) Enable UIC option"

#### Scenario: Parameter out of reasonable range
- **GIVEN** user sets `R1=1e-12` (unrealistic)
- **WHEN** validating
- **THEN** system SHALL warn: "R1=1e-12Ω is unusually small. Did you mean 1e12Ω (1TΩ)? Typical range: 1Ω - 1GΩ"

### Requirement: Debugging and Dry-Run
The system SHALL provide tools for debugging circuits before simulation.

#### Scenario: Dry-run validation
- **GIVEN** circuit configuration
- **WHEN** calling `sim.run(..., dry_run=True)`
- **THEN** system SHALL validate circuit, check engine availability, preview netlist, but not execute simulation

#### Scenario: Verbose execution logging
- **GIVEN** simulation with `verbose=True`
- **WHEN** running
- **THEN** system SHALL log: "Validating circuit... ✓", "Writing netlist... ✓", "Invoking ngspice... ✓", "Parsing results... 12,450 points"

#### Scenario: Netlist preview
- **GIVEN** circuit ready to simulate
- **WHEN** calling `circuit.preview_netlist(engine="ngspice")`
- **THEN** system SHALL display formatted netlist with syntax highlighting

### Requirement: Interactive Help
The system SHALL provide context-sensitive help and documentation.

#### Scenario: Context-sensitive help
- **GIVEN** Circuit object
- **WHEN** calling `circuit.help()`
- **THEN** system SHALL display: available methods, common workflows, relevant examples

#### Scenario: Inline examples in docstrings
- **GIVEN** any public API function
- **WHEN** viewing docstring
- **THEN** docstring SHALL include runnable example code

#### Scenario: Tutorial mode
- **GIVEN** new user
- **WHEN** running `spicelab tutorial start`
- **THEN** system SHALL launch interactive tutorial guiding through: create circuit → simulate → measure → plot

### Requirement: Type Hints and IDE Support
The system SHALL provide comprehensive type hints for excellent IDE integration.

#### Scenario: Full type coverage
- **GIVEN** any public API
- **WHEN** checked with mypy --strict
- **THEN** all types SHALL validate without errors

#### Scenario: IDE autocomplete
- **GIVEN** user typing in VS Code/PyCharm
- **WHEN** accessing Circuit.
- **THEN** IDE SHALL show all methods with parameter hints and return types

#### Scenario: Runtime type validation
- **GIVEN** function with type hints
- **WHEN** called with wrong type
- **THEN** system SHALL raise clear TypeError: "Expected Circuit, got str. Did you forget to create Circuit object?"

### Requirement: Common Patterns Library
The system SHALL provide reusable circuit patterns and building blocks.

#### Scenario: Apply compensation network pattern
- **GIVEN** opamp circuit needing frequency compensation
- **WHEN** applying pattern: `circuit.apply_pattern("miller_compensation", node_in="inp", node_out="out")`
- **THEN** system SHALL add Miller capacitor with calculated value

#### Scenario: Insert current mirror pattern
- **GIVEN** circuit needing bias current
- **WHEN** inserting pattern: `circuit.insert_pattern("current_mirror", i_ref=10e-6, ratio=2)`
- **THEN** system SHALL add current mirror with specified ratio

### Requirement: Progress Indication
The system SHALL provide progress feedback for long-running operations.

#### Scenario: Progress bar for Monte Carlo
- **GIVEN** Monte Carlo with 1000 trials
- **WHEN** executing
- **THEN** system SHALL display progress bar: "[=========>        ] 450/1000 (45%) ETA: 2m 15s"

#### Scenario: Progress for parameter sweep
- **GIVEN** grid sweep with 500 combinations
- **WHEN** running
- **THEN** system SHALL update: "Completed: 127/500 | Running: 8/8 | Cached: 23 | Failed: 2"
