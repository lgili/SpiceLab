# Mixed-Signal Co-Simulation Specification

## ADDED Requirements

### Requirement: XSPICE CodeModel Integration
The system SHALL support XSPICE behavioral models for mixed-signal simulation.

#### Scenario: Load C-based CodeModel
- **GIVEN** XSPICE CodeModel compiled as shared library
- **WHEN** circuit references model
- **THEN** ngspice SHALL load library and instantiate behavioral model

#### Scenario: ADC behavioral model
- **GIVEN** XSPICE ADC model with resolution and clock parameters
- **WHEN** simulating analog input to digital output
- **THEN** model SHALL quantize analog signal to digital codes

### Requirement: HDL Co-Simulation (Experimental)
The system SHALL support experimental co-simulation with Verilog digital blocks.

#### Scenario: Analog-to-digital interface
- **GIVEN** analog comparator in SPICE, digital logic in Verilog
- **WHEN** co-simulating
- **THEN** system SHALL exchange signals at comparator output with synchronized timesteps

#### Scenario: Clock domain synchronization
- **GIVEN** SPICE continuous time, Verilog discrete events
- **WHEN** advancing simulation
- **THEN** system SHALL synchronize time bases and handle asynchronous clocking

### Requirement: FMI/FMU Support
The system SHALL support Functional Mock-up Interface for multi-domain coupling.

#### Scenario: Load Modelica thermal FMU
- **GIVEN** thermal model exported as FMU
- **WHEN** coupling with circuit power dissipation
- **THEN** system SHALL exchange heat and temperature at each timestep

#### Scenario: Simulink co-simulation
- **GIVEN** control algorithm exported from Simulink as FMU
- **WHEN** interfacing with power circuit
- **THEN** system SHALL run closed-loop simulation with controller

### Requirement: Firmware-in-Loop API
The system SHALL provide deterministic stepping for firmware co-simulation.

#### Scenario: Sample ADC value from firmware
- **GIVEN** Python firmware function sampling V(adc_in)
- **WHEN** callback invoked at ADC clock edge
- **THEN** system SHALL provide snapshot of analog value at exact timestamp

#### Scenario: Update DAC output from firmware
- **GIVEN** firmware computing DAC code
- **WHEN** writing to DAC output
- **THEN** system SHALL update voltage source in SPICE netlist for next step
