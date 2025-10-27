# M7 - Measurement Library

## Why
SpiceLab currently lacks a comprehensive library of measurement functions for analyzing simulation results. Users need standardized, reusable measurements for AC analysis (gain, phase margin, bandwidth), transient analysis (overshoot, settling time, rise/fall), and noise analysis (SNR, THD, ENOB). This prevents efficient post-processing and automated design validation.

## What Changes
- Add `spicelab/analysis/measure.py` with typed measurement specifications
- Implement AC measurements: magnitude/phase at frequency, gain/phase crossover, GBW, phase margin (PM), gain margin (GM), fT
- Implement TRAN measurements: overshoot, undershoot, rise/fall time, settling time, SNR, THD, ENOB
- Implement NOISE measurements: input-referred noise, output noise PSD, noise figure (NF)
- Add `spicelab/analysis/signal.py` with DSP utilities: windowing, FFT, filtering
- Create typed `MeasureSpec` API with validation
- Integrate measurements with orchestrator for Grid/MC sweeps
- Add automatic Parquet export of measurement results

## Impact
- **Affected specs**: measurement, signal-processing, orchestrator
- **Affected code**:
  - New: `spicelab/analysis/measure.py`, `spicelab/analysis/signal.py`
  - Modified: `spicelab/orchestrator.py` (measurement integration)
  - New: `tests/test_measurements.py`, `examples/measurements/`
- **Dependencies**: NumPy, SciPy (signal processing), Polars (DataFrame export)
