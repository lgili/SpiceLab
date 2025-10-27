# Measurement Library Specification

## ADDED Requirements

### Requirement: Measurement Specification API
The system SHALL provide a typed `MeasureSpec` interface for defining measurements on simulation results.

#### Scenario: Define AC gain measurement
- **GIVEN** an AC analysis dataset with frequency and magnitude data
- **WHEN** user creates `GainAtFrequency(freq=1e6, signal="V(out)")`
- **THEN** the spec SHALL validate frequency is positive and signal name is valid

#### Scenario: Define settling time measurement
- **GIVEN** a transient analysis dataset
- **WHEN** user creates `SettlingTime(signal="V(out)", target=1.8, tolerance=0.05)`
- **THEN** the spec SHALL validate target and tolerance are positive numbers

### Requirement: AC Frequency Domain Measurements
The system SHALL compute standard AC analysis measurements from xarray.Dataset results.

#### Scenario: Measure magnitude and phase at frequency
- **GIVEN** AC analysis results with frequency sweep
- **WHEN** user requests magnitude and phase at 1 MHz
- **THEN** system SHALL interpolate to exact frequency and return magnitude in dB and phase in degrees

#### Scenario: Calculate gain-bandwidth product (GBW)
- **GIVEN** operational amplifier AC response
- **WHEN** user requests GBW measurement
- **THEN** system SHALL find DC gain and unity-gain frequency, return GBW = gain × f_unity

#### Scenario: Measure phase margin (PM)
- **GIVEN** AC loop gain response
- **WHEN** user requests phase margin
- **THEN** system SHALL find gain crossover frequency (0 dB) and return phase margin as 180° + phase at crossover

#### Scenario: Measure gain margin (GM)
- **GIVEN** AC loop gain response
- **WHEN** user requests gain margin
- **THEN** system SHALL find phase crossover (-180°) and return gain margin as magnitude at phase crossover in dB

### Requirement: Transient Time Domain Measurements
The system SHALL compute standard transient analysis measurements from time-series data.

#### Scenario: Measure overshoot percentage
- **GIVEN** step response with settling to steady state
- **WHEN** user requests overshoot measurement
- **THEN** system SHALL return (peak - final) / final × 100%

#### Scenario: Measure rise time (10%-90%)
- **GIVEN** rising edge signal
- **WHEN** user requests rise time with 10%-90% thresholds
- **THEN** system SHALL return time between 10% and 90% of final value

#### Scenario: Measure settling time with tolerance
- **GIVEN** transient response
- **WHEN** user requests settling time with ±5% tolerance
- **THEN** system SHALL return time when signal enters and stays within tolerance band

#### Scenario: Calculate signal-to-noise ratio (SNR)
- **GIVEN** time-domain signal with noise
- **WHEN** user requests SNR measurement
- **THEN** system SHALL compute RMS signal power and RMS noise power, return 20×log10(signal/noise) in dB

#### Scenario: Calculate total harmonic distortion (THD)
- **GIVEN** periodic signal with harmonics
- **WHEN** user requests THD
- **THEN** system SHALL perform FFT, compute THD = sqrt(sum(harmonic_powers)) / fundamental_power

#### Scenario: Calculate ENOB for ADC
- **GIVEN** ADC output with quantization
- **WHEN** user requests ENOB (effective number of bits)
- **THEN** system SHALL compute ENOB = (SINAD - 1.76) / 6.02 from SNR measurement

### Requirement: Noise Analysis Measurements
The system SHALL compute noise-related measurements from noise analysis results.

#### Scenario: Integrate noise PSD over bandwidth
- **GIVEN** noise analysis with PSD vs frequency
- **WHEN** user requests integrated noise from 10 Hz to 10 kHz
- **THEN** system SHALL integrate PSD over frequency range and return RMS noise

#### Scenario: Calculate input-referred noise
- **GIVEN** output noise and gain measurements
- **WHEN** user requests input-referred noise
- **THEN** system SHALL divide output noise by gain at each frequency

#### Scenario: Calculate noise figure (NF)
- **GIVEN** input and output noise measurements
- **WHEN** user requests noise figure
- **THEN** system SHALL compute NF = 10×log10(SNR_in / SNR_out) in dB

### Requirement: Signal Processing Utilities
The system SHALL provide DSP utilities for post-processing simulation data.

#### Scenario: Apply window function to time series
- **GIVEN** time-domain signal for FFT
- **WHEN** user applies Hann window
- **THEN** system SHALL multiply signal by Hann coefficients matching signal length

#### Scenario: Perform coherent FFT
- **GIVEN** periodic signal with known fundamental
- **WHEN** user requests coherent FFT with exact period match
- **THEN** system SHALL ensure integer number of periods and return frequency-domain representation

#### Scenario: Decimate signal for faster processing
- **GIVEN** oversampled time series
- **WHEN** user requests 10× decimation with anti-aliasing filter
- **THEN** system SHALL apply lowpass filter and downsample by factor of 10

### Requirement: Measurement Computation API
The system SHALL provide a unified API for computing measurements on datasets.

#### Scenario: Compute single measurement
- **GIVEN** xarray.Dataset from AC simulation
- **WHEN** user calls `meas.compute(ds, [PhaseMargin(signal="loop_gain")])`
- **THEN** system SHALL return polars.DataFrame with columns: measurement_name, value, unit

#### Scenario: Compute multiple measurements in batch
- **GIVEN** xarray.Dataset from transient simulation
- **WHEN** user provides list of 5 different MeasureSpec objects
- **THEN** system SHALL compute all measurements and return DataFrame with one row per measurement

#### Scenario: Preserve measurement metadata
- **GIVEN** measurement computation on dataset with sweep parameters
- **WHEN** measurements are computed
- **THEN** result DataFrame SHALL include sweep parameter columns (e.g., R_value, C_value)

### Requirement: Measurement Integration with Orchestrator
The system SHALL integrate measurements with sweep orchestration for automated analysis.

#### Scenario: Auto-compute measurements for parameter sweep
- **GIVEN** Grid sweep with 100 parameter combinations
- **WHEN** user specifies measurements to compute per run
- **THEN** orchestrator SHALL compute measurements for each combination and save to Parquet

#### Scenario: Monte Carlo with measurement statistics
- **GIVEN** Monte Carlo with 1000 trials
- **WHEN** measurements are computed per trial
- **THEN** system SHALL provide statistics: mean, std, min, max, percentiles for each measurement

### Requirement: Measurement Result Export
The system SHALL export measurement results in standard formats.

#### Scenario: Export to Parquet with compression
- **GIVEN** DataFrame with measurement results
- **WHEN** user exports to Parquet
- **THEN** system SHALL save with zstd compression and preserve metadata in footer

#### Scenario: Export to CSV for spreadsheet tools
- **GIVEN** DataFrame with measurement results
- **WHEN** user exports to CSV
- **THEN** system SHALL write CSV with headers and proper float formatting

### Requirement: Measurement Validation and Error Handling
The system SHALL validate measurement inputs and provide clear error messages.

#### Scenario: Detect missing signal in dataset
- **GIVEN** measurement spec for signal "V(out)"
- **WHEN** dataset does not contain "V(out)"
- **THEN** system SHALL raise error: "Signal 'V(out)' not found in dataset. Available: ['V(in)', 'I(R1)']"

#### Scenario: Detect insufficient data for measurement
- **GIVEN** settling time measurement requiring 10× settling periods
- **WHEN** simulation time is too short
- **THEN** system SHALL raise error: "Insufficient data: need at least {required} seconds, got {actual}"

#### Scenario: Handle numerical edge cases
- **GIVEN** phase margin measurement on unstable system (gain > 0 dB at all frequencies)
- **WHEN** computation encounters no crossover
- **THEN** system SHALL return `None` or `NaN` with warning: "No gain crossover found, system may be unstable"
