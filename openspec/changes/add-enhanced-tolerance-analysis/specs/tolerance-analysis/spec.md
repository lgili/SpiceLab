## ADDED Requirements

### Requirement: Correlated Parameter Groups

The system SHALL support grouping parameters that vary together (correlated variation) in Monte Carlo analysis.

#### Scenario: Same-lot resistors vary together
- **GIVEN** a circuit with resistors R1, R2, R3 from the same manufacturing lot
- **WHEN** Monte Carlo analysis is run with `CorrelatedGroup([R1, R2, R3], NormalPct(0.1))`
- **THEN** all three resistors SHALL receive the same random multiplier in each iteration

#### Scenario: Mixed correlated and independent parameters
- **GIVEN** a circuit with correlated resistors and independent op-amp offsets
- **WHEN** Monte Carlo is configured with both `CorrelatedGroup` and individual distributions
- **THEN** correlated parameters SHALL share random factors while independent parameters SHALL have separate random factors

#### Scenario: Multiple correlation groups
- **GIVEN** a circuit with two sets of matched components (e.g., input stage and output stage)
- **WHEN** two separate `CorrelatedGroup` instances are provided
- **THEN** each group SHALL have its own independent random factor

---

### Requirement: Absolute Tolerance Distributions

The system SHALL support absolute value tolerances (not percentage-based) for parameters like voltage offsets.

#### Scenario: Normal distribution with absolute sigma
- **GIVEN** an op-amp with ±2mV input offset voltage (3-sigma)
- **WHEN** `NormalAbs(0.002)` distribution is applied
- **THEN** sampled values SHALL follow N(0, 0.002/3) distribution in volts

#### Scenario: Triangular distribution with absolute range
- **GIVEN** a parameter with ±5mV uniform-like variation
- **WHEN** `TriangularAbs(0.005)` distribution is applied
- **THEN** sampled values SHALL follow triangular distribution centered at 0 with half-width 0.005

#### Scenario: Absolute tolerance in WCA
- **GIVEN** a WCA configuration with `{Voff: (0.002, 'abs')}`
- **WHEN** WCA corners are generated
- **THEN** corners SHALL be nominal ± 0.002 (not nominal × (1 ± 0.002))

---

### Requirement: Temperature Sweep Integration

The system SHALL provide integrated multi-temperature Monte Carlo and WCA analysis.

#### Scenario: Monte Carlo across temperature range
- **GIVEN** a circuit and temperature list `[-40, 25, 85]`
- **WHEN** `monte_carlo_sweep(circuit, tolerances, temperatures=[...], n_iterations=1000)` is called
- **THEN** the system SHALL run 1000 iterations at each temperature and return combined results

#### Scenario: Access results by temperature
- **GIVEN** a `MultiConditionResult` from temperature sweep
- **WHEN** `result.at_condition(temperature=25)` is called
- **THEN** only results for T=25°C SHALL be returned

#### Scenario: DataFrame with temperature index
- **GIVEN** a `MultiConditionResult` from temperature sweep
- **WHEN** `result.to_dataframe()` is called
- **THEN** DataFrame SHALL have multi-index with (temperature, iteration) levels

#### Scenario: WCA across temperatures
- **GIVEN** a circuit and temperature list
- **WHEN** `wca_sweep(circuit, tolerances, temperatures=[...])` is called
- **THEN** all WCA corners SHALL be evaluated at each temperature

---

### Requirement: Mixed Tolerance WCA

The system SHALL support mixing relative and absolute tolerances in WCA analysis.

#### Scenario: Relative and absolute tolerances together
- **GIVEN** WCA configuration `{R1: 0.001, Voff: (0.002, 'abs')}`
- **WHEN** WCA corners are generated
- **THEN** R1 corners SHALL be `R1_nom × (1 ± 0.001)` AND Voff corners SHALL be `0 ± 0.002`

#### Scenario: Backward compatibility with relative-only
- **GIVEN** WCA configuration `{R1: 0.001, R2: 0.001}` (existing format)
- **WHEN** WCA is executed
- **THEN** behavior SHALL be identical to previous implementation

---

### Requirement: Simplified Result Extraction

The system SHALL provide convenient methods to extract common values from simulation results.

#### Scenario: Extract node voltage
- **GIVEN** a simulation result with node 'vout'
- **WHEN** `result.get_voltage('vout')` is called
- **THEN** the final voltage value at node 'vout' SHALL be returned as float

#### Scenario: Case-insensitive node names
- **GIVEN** a simulation with node 'Vout' (capitalized)
- **WHEN** `result.get_voltage('vout')` is called (lowercase)
- **THEN** the value SHALL be returned without error

#### Scenario: Missing node error
- **GIVEN** a simulation result
- **WHEN** `result.get_voltage('nonexistent')` is called
- **THEN** a descriptive error SHALL be raised listing available nodes

#### Scenario: Extract component current
- **GIVEN** a simulation result with current through R1
- **WHEN** `result.get_current('R1')` is called
- **THEN** the current through R1 SHALL be returned as float

---

### Requirement: Process Capability Metrics

The system SHALL calculate industry-standard process capability metrics from Monte Carlo results.

#### Scenario: Calculate Cpk
- **GIVEN** Monte Carlo results with temperature error data
- **WHEN** `result.cpk(metric='temp_error', lsl=-0.5, usl=0.5)` is called
- **THEN** Cpk SHALL be calculated as `min((USL-μ)/(3σ), (μ-LSL)/(3σ))`

#### Scenario: Estimate yield
- **GIVEN** Monte Carlo results
- **WHEN** `result.yield_estimate(metric='temp_error', lsl=-0.5, usl=0.5)` is called
- **THEN** estimated yield percentage SHALL be returned based on normal distribution

#### Scenario: Calculate sigma level
- **GIVEN** Monte Carlo results with known Cpk
- **WHEN** `result.sigma_level(metric='temp_error', lsl=-0.5, usl=0.5)` is called
- **THEN** equivalent sigma level (e.g., 3σ, 6σ) SHALL be returned

#### Scenario: Process summary
- **GIVEN** Monte Carlo results
- **WHEN** `result.process_summary(metric='temp_error', spec=(-0.5, 0.5))` is called
- **THEN** a summary dict with mean, std, Cpk, yield, sigma_level SHALL be returned

---

### Requirement: Automated Report Generation

The system SHALL generate comprehensive tolerance analysis reports.

#### Scenario: Generate HTML report
- **GIVEN** Monte Carlo or WCA results
- **WHEN** `result.generate_report('report.html')` is called
- **THEN** an HTML file SHALL be created with histograms, tables, and metrics

#### Scenario: Report includes histograms
- **GIVEN** Monte Carlo results
- **WHEN** report is generated
- **THEN** histogram plots with spec limits overlay SHALL be included

#### Scenario: Report includes Cpk table
- **GIVEN** Monte Carlo results with multiple metrics
- **WHEN** report is generated
- **THEN** a summary table with Cpk, yield, sigma level per metric SHALL be included

#### Scenario: Report includes worst corners
- **GIVEN** WCA results
- **WHEN** report is generated
- **THEN** worst-case corner combinations SHALL be listed with their values

#### Scenario: Generate PDF report
- **GIVEN** Monte Carlo results and weasyprint installed
- **WHEN** `result.generate_report('report.pdf')` is called
- **THEN** a PDF file SHALL be created with same content as HTML
