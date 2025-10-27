# Interactive Widgets Specification

## ADDED Requirements

### Requirement: Parameter Slider Widget
The system SHALL provide interactive parameter slider widgets for Jupyter notebooks.

#### Scenario: Create parameter slider
- **GIVEN** circuit with parameter `R1`
- **WHEN** user creates `ParamSlider(circuit, "R1", min=100, max=10000, step=100)`
- **THEN** widget SHALL display slider with current value and unit

#### Scenario: Live simulation on slider change
- **GIVEN** parameter slider connected to circuit
- **WHEN** user moves slider to new value
- **THEN** system SHALL rerun simulation automatically and update plots

#### Scenario: Multiple parameter control
- **GIVEN** circuit with `[R1, C1, L1]`
- **WHEN** user creates multi-slider widget
- **THEN** widget SHALL show all three sliders with synchronized updates

### Requirement: Interactive Plot Widgets
The system SHALL provide interactive plot widgets with measurement cursors and annotations.

#### Scenario: Bode plot with cursors
- **GIVEN** AC analysis results
- **WHEN** user displays Bode widget
- **THEN** widget SHALL show magnitude and phase with draggable cursors displaying exact values

#### Scenario: Annotate measurements on plot
- **GIVEN** Bode plot widget showing loop gain
- **WHEN** widget computes phase margin
- **THEN** widget SHALL overlay PM annotation: vertical line at crossover freq with label "PM = 65°"

#### Scenario: Step response with settling time
- **GIVEN** transient step response
- **WHEN** displaying step widget
- **THEN** widget SHALL show settling time bands (±5%) and annotate settling time value

### Requirement: Widget State Persistence
The system SHALL support saving and restoring widget states.

#### Scenario: Export widget parameters
- **GIVEN** configured parameter sliders
- **WHEN** user exports widget state
- **THEN** system SHALL save parameter values and ranges to JSON

#### Scenario: Restore widget from saved state
- **GIVEN** saved widget state file
- **WHEN** user loads widget
- **THEN** widget SHALL restore all parameter values and recreate plots
