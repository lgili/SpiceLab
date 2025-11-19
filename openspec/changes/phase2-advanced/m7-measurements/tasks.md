# M7: Measurement & Analysis Library - Tasks

**Status:** Proposed
**Start Date:** TBD
**Target Completion:** TBD (8-10 weeks)

## Task Breakdown

### Phase 1: Framework and Infrastructure (Week 1-2)
- [ ] Create `spicelab/measurements/` package structure
  - [ ] Add `__init__.py` with exports
  - [ ] Create `base.py` with protocols
  - [ ] Create `registry.py` for measurement registration
  - [ ] Create subdirectories: `ac/`, `transient/`, `spectrum/`, `digital/`, `power/`
- [ ] Implement base `Measurement` protocol
  - [ ] Define `MeasurementResult` Pydantic model
  - [ ] Create `Measurement` Protocol with `measure()` and `check_spec()` methods
  - [ ] Add type hints and docstrings
- [ ] Create `MeasurementRegistry`
  - [ ] Implement decorator-based registration
  - [ ] Add `get()`, `list_all()`, `search()` methods
  - [ ] Support aliases and categories
- [ ] Build `MeasurementResult` data model
  - [ ] Add `value`, `unit`, `passed`, `metadata` fields
  - [ ] Implement JSON serialization
  - [ ] Add comparison operators
- [ ] Implement DSL for custom measurements
  - [ ] Create `MeasurementDSL.define()` factory
  - [ ] Support lambda/function-based measurements
  - [ ] Add validation for custom measurements
- [ ] Write framework unit tests
  - [ ] Test registry registration
  - [ ] Test result serialization
  - [ ] Test DSL measurement creation

**Estimated Time:** 2 weeks
**Assignee:** TBD

---

### Phase 2: AC Analysis Measurements (Week 3-4)
- [ ] Implement gain measurement
  - [ ] DC gain (from .op or AC at DC)
  - [ ] AC gain at specific frequency
  - [ ] Unit conversion (dB, V/V, abs)
  - [ ] Spec checking (min/max)
- [ ] Create bandwidth measurement
  - [ ] -3dB bandwidth calculation
  - [ ] Auto-detect DC gain
  - [ ] Configurable -NdB point
  - [ ] Handle cases where -3dB not reached
- [ ] Add phase margin measurement
  - [ ] Find unity-gain frequency (0dB crossing)
  - [ ] Calculate phase at UGF
  - [ ] Stability checking (PM > 0)
  - [ ] Handle multiple crossings
- [ ] Implement gain margin measurement
  - [ ] Find -180° phase crossing
  - [ ] Calculate gain at crossing
  - [ ] Stability checking (GM > 0)
- [ ] Create input/output impedance measurements
  - [ ] Zin calculation from AC
  - [ ] Zout calculation from AC
  - [ ] Frequency-dependent impedance
- [ ] Add PSRR measurement
  - [ ] Calculate from AC simulation
  - [ ] Support multiple frequencies
  - [ ] dB conversion
- [ ] Implement CMRR measurement
  - [ ] Differential gain measurement
  - [ ] Common-mode gain measurement
  - [ ] CMRR = Adiff / Acm
  - [ ] Document required simulation setup
- [ ] Write tests for all AC measurements
  - [ ] Unit tests with synthetic data
  - [ ] Integration tests with real circuits
  - [ ] Edge case testing

**Estimated Time:** 2 weeks
**Assignee:** TBD

---

### Phase 3: Transient Analysis Measurements (Week 5-6)
- [ ] Implement slew rate measurement
  - [ ] Rising edge slew rate
  - [ ] Falling edge slew rate
  - [ ] Configurable thresholds (10%-90%)
  - [ ] Handle multiple edges
- [ ] Create settling time measurement
  - [ ] Auto-detect final value
  - [ ] Configurable error band (%)
  - [ ] Find last crossing
  - [ ] Handle non-settling cases
- [ ] Add overshoot/undershoot measurement
  - [ ] Peak overshoot detection
  - [ ] Undershoot detection
  - [ ] Percentage calculation
  - [ ] Spec checking
- [ ] Implement rise/fall time measurement
  - [ ] 10%-90% rise time
  - [ ] 90%-10% fall time
  - [ ] Configurable thresholds
  - [ ] Multiple edge handling
- [ ] Create propagation delay measurement
  - [ ] Input-to-output delay
  - [ ] Rising/falling edge delays
  - [ ] 50% threshold crossing
  - [ ] Handle multiple stages
- [ ] Write tests for transient measurements
  - [ ] Unit tests with step responses
  - [ ] Integration tests with real circuits
  - [ ] Edge case testing (noise, glitches)

**Estimated Time:** 2 weeks
**Assignee:** TBD

---

### Phase 4: Spectrum and Distortion Analysis (Week 7)
- [ ] Implement FFT helper with windowing
  - [ ] Support windows: Hann, Hamming, Blackman, Flattop
  - [ ] Frequency axis generation
  - [ ] Positive frequency extraction
  - [ ] Power spectrum calculation
- [ ] Create THD measurement
  - [ ] FFT-based harmonic extraction
  - [ ] Configurable number of harmonics
  - [ ] THD% and THD dB
  - [ ] Fundamental frequency detection
- [ ] Add THD+N measurement
  - [ ] Total harmonic distortion + noise
  - [ ] Noise floor calculation
  - [ ] Band-limited measurement
- [ ] Implement SNR/SINAD measurements
  - [ ] Signal-to-noise ratio
  - [ ] Signal-to-noise-and-distortion
  - [ ] RMS-based or FFT-based
  - [ ] Configurable noise bands
- [ ] Create SFDR measurement
  - [ ] Spurious-free dynamic range
  - [ ] Find largest spurious tone
  - [ ] Exclude harmonics option
- [ ] Add spectrum plotting utilities
  - [ ] Magnitude spectrum plot
  - [ ] Phase spectrum plot
  - [ ] Logarithmic axes
  - [ ] Marker annotations
- [ ] Write tests for spectrum analysis
  - [ ] Synthetic sine waves with harmonics
  - [ ] Known THD values
  - [ ] Window function validation

**Estimated Time:** 1 week
**Assignee:** TBD

---

### Phase 5: Digital Signal Measurements (Week 8)
- [ ] Implement eye diagram generation
  - [ ] UI (unit interval) extraction
  - [ ] Resampling to common time grid
  - [ ] Trace overlay
  - [ ] Configurable samples per UI
- [ ] Create eye opening measurement
  - [ ] Vertical eye opening (height)
  - [ ] Horizontal eye opening (width)
  - [ ] Sample point selection
  - [ ] High/low level separation
- [ ] Add jitter measurement
  - [ ] Period jitter (RMS)
  - [ ] Cycle-to-cycle jitter
  - [ ] Time interval error (TIE)
  - [ ] Edge detection (rising/falling)
  - [ ] Threshold crossing
- [ ] Implement bit error rate estimation
  - [ ] Eye opening correlation
  - [ ] Bathtub curve generation
  - [ ] BER extrapolation
- [ ] Create timing measurements
  - [ ] Setup time measurement
  - [ ] Hold time measurement
  - [ ] Clock-to-Q delay
- [ ] Write tests for digital measurements
  - [ ] Synthetic digital waveforms
  - [ ] Known jitter patterns
  - [ ] Eye diagram validation

**Estimated Time:** 1 week
**Assignee:** TBD

---

### Phase 6: Power Integrity (Week 9)
- [ ] Implement PDN impedance measurement
  - [ ] Z(f) from AC analysis
  - [ ] Find maximum impedance
  - [ ] Frequency at max Z
  - [ ] Target impedance checking
  - [ ] Document simulation setup (AC current source)
- [ ] Create supply ripple measurement
  - [ ] Peak-to-peak ripple
  - [ ] FFT-based ripple at frequency
  - [ ] RMS ripple
  - [ ] Spec checking
- [ ] Add load transient response
  - [ ] Voltage droop measurement
  - [ ] Recovery time
  - [ ] Undershoot/overshoot
  - [ ] Spec compliance
- [ ] Implement voltage droop measurement
  - [ ] Maximum droop detection
  - [ ] Droop percentage
  - [ ] Time at max droop
- [ ] Create efficiency calculation
  - [ ] Pout / Pin
  - [ ] Support for multiple supplies
  - [ ] Time-averaged efficiency
- [ ] Write tests for power measurements
  - [ ] Synthetic PDN impedance data
  - [ ] Load step responses
  - [ ] Known efficiency circuits

**Estimated Time:** 1 week
**Assignee:** TBD

---

### Phase 7: Documentation and Integration (Week 10)
- [ ] Document all 30+ measurements
  - [ ] API reference for each measurement
  - [ ] Parameter descriptions
  - [ ] Return value documentation
  - [ ] Required simulation setup
- [ ] Create measurement cookbook
  - [ ] Common circuit examples (opamp, LDO, ADC, etc.)
  - [ ] Measurement combinations
  - [ ] Spec checking examples
  - [ ] Best practices
- [ ] Add measurement chaining/composition examples
  - [ ] Multiple measurements on single dataset
  - [ ] Derived measurements
  - [ ] Conditional measurements
- [ ] Write integration tests
  - [ ] Full circuit simulations
  - [ ] Multi-measurement validation
  - [ ] Error handling tests
  - [ ] Performance tests
- [ ] Create performance benchmarks
  - [ ] Measurement extraction time
  - [ ] FFT performance (various sizes)
  - [ ] Eye diagram generation speed
  - [ ] Memory usage
- [ ] Update main documentation
  - [ ] Add measurements section to docs
  - [ ] Tutorial on using measurements
  - [ ] Migration guide (if applicable)
  - [ ] Add to API reference

**Estimated Time:** 1 week
**Assignee:** TBD

---

## Acceptance Criteria

### Must Have
- [ ] 30+ measurements implemented
  - [ ] AC analysis: gain, bandwidth, phase margin, gain margin, PSRR, CMRR, impedance (7)
  - [ ] Transient: slew rate, settling time, overshoot, rise/fall time, delay (5)
  - [ ] Spectrum: THD, THD+N, SNR, SINAD, SFDR (5)
  - [ ] Digital: eye diagram, eye opening, jitter, BER, timing (5)
  - [ ] Power: PDN impedance, ripple, droop, efficiency, transient (5)
  - [ ] Additional: noise, crosstalk, distortion analysis (3+)
- [ ] Measurement framework complete
  - [ ] Registry with registration decorator
  - [ ] Result data model with serialization
  - [ ] Spec checking built-in
  - [ ] Custom measurement DSL
- [ ] Test coverage ≥95%
- [ ] All measurements documented
- [ ] Performance targets met
  - [ ] <100ms per measurement
  - [ ] <500ms FFT (100k points)
  - [ ] <1s eye diagram (1M points)

### Should Have
- [ ] Measurement chaining/composition
- [ ] Automatic unit conversion
- [ ] Batch measurement runner
- [ ] Measurement result export (JSON, CSV)
- [ ] Plotting utilities for each measurement type

### Nice to Have
- [ ] Interactive measurement explorer (web UI)
- [ ] Measurement templates for common circuits
- [ ] Measurement uncertainty estimation
- [ ] Statistical analysis across Monte Carlo runs

## Testing Checklist

Before marking M7 as complete:
- [ ] Run full test suite (unit + integration)
- [ ] Verify all 30+ measurements work on real circuits
- [ ] Validate against manual calculations (±1% tolerance)
- [ ] Performance benchmarks pass targets
- [ ] Documentation complete with examples
- [ ] Code review by peer
- [ ] User acceptance testing (if possible)

## Notes

- **Window Functions:** Document when to use each window type (Hann for general, Flattop for amplitude accuracy, etc.)
- **Interpolation:** Use scipy.interpolate for robust interpolation when finding crossings
- **FFT Resolution:** Warn users about frequency resolution limitations (Δf = 1/T)
- **CMRR/PSRR:** Require specific simulation setups - provide templates
- **Jitter:** Multiple jitter metrics available - document when to use each

## Dependencies

- M1 (testing framework for validation)
- M2 (performance baseline for benchmarks)
- M4 (error handling for robust measurements)
- scipy (signal processing, FFT)
- numpy (numerical operations)
- xarray (dataset handling)

## Blocking

- M9 (Optimization) will use measurements as objectives
- M11 (UX) will use measurements for visualization

---

**Last Updated:** 2025-01-19
