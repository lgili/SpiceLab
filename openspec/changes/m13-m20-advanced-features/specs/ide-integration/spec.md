# IDE Integration & Schematic I/O Specification

## ADDED Requirements

### Requirement: VS Code Extension
The system SHALL provide VS Code extension for integrated circuit simulation workflow.

#### Scenario: Run simulation from editor
- **GIVEN** circuit file open in VS Code
- **WHEN** user clicks "Run Simulation" or uses keyboard shortcut
- **THEN** extension SHALL execute simulation and show results in panel

#### Scenario: Live plot updates
- **GIVEN** running simulation
- **WHEN** results arrive
- **THEN** extension SHALL update plots in real-time without page refresh

#### Scenario: Signal inspector on hover
- **GIVEN** netlist in editor
- **WHEN** user hovers over signal name like `V(out)`
- **THEN** extension SHALL show tooltip with current value and waveform preview

### Requirement: LTspice Schematic Import/Export
The system SHALL support bidirectional conversion between LTspice .asc files and Circuit objects.

#### Scenario: Parse LTspice .asc to Circuit
- **GIVEN** LTspice schematic file `opamp.asc`
- **WHEN** importing with `Circuit.from_ltspice("opamp.asc")`
- **THEN** system SHALL parse components, nets, and parameters into Circuit object

#### Scenario: Export Circuit to .asc
- **GIVEN** Circuit object created programmatically
- **WHEN** exporting with `circuit.to_ltspice("output.asc")`
- **THEN** system SHALL generate valid .asc file openable in LTspice

#### Scenario: Round-trip preservation
- **GIVEN** original .asc file
- **WHEN** importing to Circuit and exporting back
- **THEN** re-exported .asc SHALL produce identical simulation results

### Requirement: KiCad Integration
The system SHALL support importing KiCad schematics for simulation.

#### Scenario: Parse KiCad schematic to netlist
- **GIVEN** KiCad .kicad_sch file
- **WHEN** importing schematic
- **THEN** system SHALL extract netlist with component values and connections

#### Scenario: Handle KiCad symbol libraries
- **GIVEN** KiCad schematic with custom symbols
- **WHEN** importing
- **THEN** system SHALL map symbols to SPICE models or prompt for mapping

### Requirement: Netlist Diff and Explain
The system SHALL provide visual diff for netlists with explanations.

#### Scenario: Diff two netlists
- **GIVEN** two versions of same circuit netlist
- **WHEN** user requests diff
- **THEN** system SHALL highlight added/removed/modified components

#### Scenario: Explain netlist line
- **GIVEN** netlist line: `R1 n1 n2 1k`
- **WHEN** user clicks "explain"
- **THEN** system SHALL display: "Resistor R1 connects nodes n1 and n2 with value 1kΩ"

### Requirement: Measurement DSL
The system SHALL support declarative measurement language similar to SPICE .meas.

#### Scenario: Parse measurement DSL
- **GIVEN** DSL string: `"MEASURE TRAN overshoot MAX V(out)-V(final) FROM 0.9ms TO 1.1ms"`
- **WHEN** parsing
- **THEN** system SHALL compile to MeasureSpec equivalent

#### Scenario: Execute DSL measurements
- **GIVEN** compiled DSL measurements
- **WHEN** simulation completes
- **THEN** system SHALL execute measurements and return results matching manual calculation
