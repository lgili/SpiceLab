# Implementation Tasks - M7 Measurement Library

## 1. Core Measurement Infrastructure
- [ ] 1.1 Create `spicelab/analysis/measure.py` with base types
- [ ] 1.2 Define `MeasureSpec` Pydantic model with validation
- [ ] 1.3 Implement `Measurement` result class with metadata
- [ ] 1.4 Create measurement registry pattern for extensibility

## 2. AC Analysis Measurements
- [ ] 2.1 Implement magnitude and phase extraction at specific frequency
- [ ] 2.2 Implement gain crossover frequency (0 dB crossing)
- [ ] 2.3 Implement phase crossover frequency (-180° crossing)
- [ ] 2.4 Implement gain-bandwidth product (GBW) calculation
- [ ] 2.5 Implement phase margin (PM) measurement
- [ ] 2.6 Implement gain margin (GM) measurement
- [ ] 2.7 Implement fT (unity gain frequency) for transistors

## 3. Transient Analysis Measurements
- [ ] 3.1 Implement overshoot and undershoot detection
- [ ] 3.2 Implement rise time and fall time (10%-90%, 20%-80%)
- [ ] 3.3 Implement settling time with configurable tolerance (ε%)
- [ ] 3.4 Implement signal-to-noise ratio (SNR) calculation
- [ ] 3.5 Implement total harmonic distortion (THD)
- [ ] 3.6 Implement effective number of bits (ENOB) for ADCs

## 4. Noise Analysis Measurements
- [ ] 4.1 Implement input-referred noise calculation
- [ ] 4.2 Implement output noise PSD integration over bandwidth
- [ ] 4.3 Implement noise figure (NF) measurement

## 5. Signal Processing Utilities
- [ ] 5.1 Create `spicelab/analysis/signal.py`
- [ ] 5.2 Implement windowing functions (Hann, Hamming, Blackman, Kaiser)
- [ ] 5.3 Implement coherent FFT with zero-padding options
- [ ] 5.4 Implement decimation and resampling
- [ ] 5.5 Implement FIR/IIR filter helpers for post-processing

## 6. Integration and API
- [ ] 6.1 Create `meas.compute(ds: xr.Dataset, specs: list[MeasureSpec]) -> pl.DataFrame`
- [ ] 6.2 Integrate with orchestrator for per-point measurement in sweeps
- [ ] 6.3 Add automatic Parquet export of measurement tables
- [ ] 6.4 Create convenience methods for common measurements

## 7. Testing and Examples
- [ ] 7.1 Create test fixtures (RC, RL, OpAmp responses)
- [ ] 7.2 Write tests for AC measurements with tolerance assertions
- [ ] 7.3 Write tests for TRAN measurements with tolerance assertions
- [ ] 7.4 Write tests for NOISE measurements
- [ ] 7.5 Create example: "Bode plot with PM/GM annotation"
- [ ] 7.6 Create example: "Step response with settling/overshoot analysis"
- [ ] 7.7 Create example: "ADC ENOB and SNR measurement"

## 8. Documentation
- [ ] 8.1 Write `docs/measurements.md` with API reference
- [ ] 8.2 Document all measurement formulas and references
- [ ] 8.3 Document windowing and FFT best practices
- [ ] 8.4 Add measurement examples to cookbook
