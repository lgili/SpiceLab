# M7: Measurement & Analysis Library - Tasks

**Status:** In Progress (Framework Complete)
**Start Date:** 2025-11-29
**Target Completion:** TBD (remaining: CMRR, noise, documentation)

## Task Breakdown

### Phase 1: Framework and Infrastructure (Week 1-2) ✅ COMPLETE
- [x] Create `spicelab/measurements/` package structure
  - [x] Add `__init__.py` with exports
  - [x] Create `base.py` with protocols
  - [x] Create `registry.py` for measurement registration
  - [x] Create modules: `ac.py`, `transient.py`, `spectrum.py`, `digital.py`, `power.py`
- [x] Implement base `Measurement` protocol
  - [x] Define `MeasurementResult` dataclass (value, unit, passed, metadata)
  - [x] Create `Measurement` Protocol with `measure()` and `check_spec()` methods
  - [x] Create `BaseMeasurement` base class with default spec checking
  - [x] Add type hints and docstrings
- [x] Create `MeasurementRegistry`
  - [x] Implement decorator-based registration (`@measurement`)
  - [x] Add `get()`, `list_all()`, `list_by_analysis()`, `info()` methods
  - [x] Support categories via required_analyses
- [x] Build `MeasurementResult` data model
  - [x] Add `value`, `unit`, `passed`, `metadata` fields
  - [x] Implement `to_dict()` serialization
  - [x] Add string representation
- [x] Implement `measure()` convenience function
  - [x] Look up measurement by name
  - [x] Support spec checking
  - [x] Pass kwargs to measurement constructor
- [x] Write framework unit tests
  - [x] Test registry registration
  - [x] Test result serialization
  - [x] Test custom measurement decorator

**Completed:** 2025-11-29

---

### Phase 2: AC Analysis Measurements (Week 3-4) ✅ COMPLETE
- [x] Implement gain measurement
  - [x] DC gain (from .op or AC at DC)
  - [x] AC gain at specific frequency
  - [x] Unit conversion (dB, V/V, abs)
  - [x] Spec checking (min/max)
- [x] Create bandwidth measurement
  - [x] -3dB bandwidth calculation
  - [x] Auto-detect DC gain
  - [x] Configurable -NdB point
  - [x] Handle cases where -3dB not reached
- [x] Add phase margin measurement
  - [x] Find unity-gain frequency (0dB crossing)
  - [x] Calculate phase at UGF
  - [x] Stability checking (PM > 0)
  - [x] Handle multiple crossings
- [x] Implement gain margin measurement
  - [x] Find -180° phase crossing
  - [x] Calculate gain at crossing
  - [x] Stability checking (GM > 0)
- [x] Create input impedance measurement
  - [x] Zin calculation from AC
  - [x] Frequency-dependent impedance
  - [x] Complex impedance (magnitude and phase)
- [x] Add PSRR measurement
  - [x] Calculate from AC simulation
  - [x] Support multiple frequencies
  - [x] dB conversion
- [x] Implement CMRR measurement
  - [x] Support pre-computed gains
  - [x] Support dataset-based measurement
  - [x] CMRR = 20*log10(Adiff / Acm)
  - [x] Document required simulation setup
- [x] Write tests for AC measurements
  - [x] Unit tests with synthetic data
  - [x] Edge case testing

**Completed:** 2025-11-29 (7/7 measurements)

---

### Phase 3: Transient Analysis Measurements (Week 5-6) ✅ COMPLETE
- [x] Implement slew rate measurement
  - [x] Rising edge slew rate
  - [x] Falling edge slew rate
  - [x] Configurable thresholds (10%-90%)
  - [x] "both" mode for average
- [x] Create settling time measurement
  - [x] Auto-detect final value (last 10%)
  - [x] Configurable error band (%)
  - [x] Find last time outside band
- [x] Add overshoot measurement
  - [x] Peak overshoot detection
  - [x] Percentage calculation
  - [x] Spec checking via base class
- [x] Implement rise time measurement
  - [x] 10%-90% rise time (default)
  - [x] Configurable thresholds
  - [x] Auto-detect reference/target
- [x] Implement fall time measurement
  - [x] 90%-10% fall time (default)
  - [x] Configurable thresholds
- [x] Create propagation delay measurement
  - [x] Input-to-output delay
  - [x] Rising/falling edge delays
  - [x] 50% threshold crossing (configurable)
- [x] Write tests for transient measurements
  - [x] Unit tests with step responses
  - [x] Edge case testing

**Completed:** 2025-11-29

---

### Phase 4: Spectrum and Distortion Analysis (Week 7) ✅ COMPLETE
- [x] Implement FFT helper with windowing
  - [x] Support windows: rect, Hann, Hamming, Blackman, Flattop
  - [x] Frequency axis generation
  - [x] Positive frequency extraction
  - [x] Proper amplitude scaling
- [x] Create THD measurement
  - [x] FFT-based harmonic extraction
  - [x] Configurable number of harmonics
  - [x] THD% and THD dB in metadata
  - [x] Fundamental frequency auto-detection
- [x] Add THD+N measurement
  - [x] Total harmonic distortion + noise
  - [x] Noise floor calculation
  - [x] Configurable bandwidth
- [x] Implement SNR measurement
  - [x] Signal-to-noise ratio
  - [x] FFT-based
  - [x] Configurable noise bands
- [x] Implement SINAD measurement
  - [x] Signal-to-noise-and-distortion
  - [x] Total power calculation
- [x] Create SFDR measurement
  - [x] Spurious-free dynamic range
  - [x] Find largest spurious tone
  - [x] Exclude fundamental
- [x] Add ENOB measurement
  - [x] Effective number of bits
  - [x] Based on SINAD
- [x] Write tests for spectrum analysis
  - [x] Synthetic sine waves with harmonics
  - [x] Window function validation

**Completed:** 2025-11-29 (6/6 measurements)

---

### Phase 5: Digital Signal Measurements (Week 8) ✅ COMPLETE
- [x] Implement eye diagram generation (`EyeDiagram` class)
  - [x] UI (unit interval) extraction
  - [x] Resampling to common time grid
  - [x] Trace overlay
  - [x] Configurable samples per UI and n_ui
  - [x] `EyeDiagramData` dataclass with histogram support
  - [x] Optional matplotlib plotting
- [x] Create eye opening measurement
  - [x] Vertical eye opening (height)
  - [x] Horizontal eye opening (width in UI)
  - [x] Sample point selection
  - [x] High/low level separation
- [x] Add jitter measurement
  - [x] Period jitter (RMS)
  - [x] Cycle-to-cycle jitter
  - [x] Peak-to-peak jitter
  - [x] Edge detection (rising/falling)
  - [x] Threshold crossing (auto or manual)
- [x] Add duty cycle measurement
  - [x] High time / period calculation
  - [x] Auto threshold detection
- [ ] Implement bit error rate estimation (pending)
  - [ ] Eye opening correlation
  - [ ] BER extrapolation
- [ ] Create timing measurements (pending)
  - [ ] Setup time measurement
  - [ ] Hold time measurement
- [x] Write tests for digital measurements
  - [x] Synthetic digital waveforms
  - [x] Known jitter patterns
  - [x] Eye diagram validation

**Completed:** 2025-11-29 (4/6 measurements)

---

### Phase 6: Power Integrity (Week 9) ✅ COMPLETE
- [x] Implement PDN impedance measurement
  - [x] Z(f) from AC analysis
  - [x] Find maximum impedance
  - [x] Frequency at max Z
  - [x] Target impedance checking (pass/fail)
- [x] Create supply ripple measurement
  - [x] Peak-to-peak ripple
  - [x] FFT-based ripple at specific frequency
  - [x] Steady-state start time filter
- [x] Add load transient response measurement
  - [x] Voltage droop measurement
  - [x] Overshoot detection
  - [x] Recovery time calculation
  - [x] Deviation type (droop/overshoot)
- [x] Create efficiency calculation
  - [x] Pout / Pin percentage
  - [x] Power loss calculation
  - [x] Steady-state filter
- [x] Implement voltage droop measurement
  - [x] Maximum droop detection
  - [x] Droop percentage
  - [x] Time window support
  - [x] Auto nominal voltage detection
- [x] Write tests for power measurements
  - [x] Synthetic PDN impedance data
  - [x] Load step responses
  - [x] Known efficiency circuits

**Completed:** 2025-11-29 (5/5 measurements)

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

**Last Updated:** 2025-11-29

## Current Measurement Inventory

| Category | Measurement | Status |
|----------|-------------|--------|
| **AC** | gain | ✅ Complete |
| | bandwidth | ✅ Complete |
| | phase_margin | ✅ Complete |
| | gain_margin | ✅ Complete |
| | psrr | ✅ Complete |
| | input_impedance | ✅ Complete |
| | cmrr | ✅ Complete |
| **Transient** | slew_rate | ✅ Complete |
| | settling_time | ✅ Complete |
| | rise_time | ✅ Complete |
| | fall_time | ✅ Complete |
| | overshoot | ✅ Complete |
| | propagation_delay | ✅ Complete |
| **Spectrum** | thd | ✅ Complete |
| | snr | ✅ Complete |
| | sinad | ✅ Complete |
| | sfdr | ✅ Complete |
| | enob | ✅ Complete |
| | thd_n | ✅ Complete |
| **Digital** | eye_opening | ✅ Complete |
| | jitter | ✅ Complete |
| | duty_cycle | ✅ Complete |
| | ber | ⏳ Pending |
| | setup_hold | ⏳ Pending |
| **Power** | ripple | ✅ Complete |
| | pdn_impedance | ✅ Complete |
| | load_transient | ✅ Complete |
| | efficiency | ✅ Complete |
| | voltage_droop | ✅ Complete |

**Total:** 27 complete, 3 pending (BER, setup/hold timing)

## Files Created

```
spicelab/measurements/
├── __init__.py          # Package exports, module imports
├── base.py              # MeasurementResult, Measurement protocol, BaseMeasurement
├── registry.py          # MeasurementRegistry, @measurement decorator, measure()
├── ac.py                # AC measurements (gain, bandwidth, margins, PSRR, impedance)
├── transient.py         # Transient measurements (slew, settling, rise/fall, overshoot)
├── spectrum.py          # Spectrum measurements (THD, SNR, SINAD, SFDR, ENOB)
├── digital.py           # Digital measurements (eye diagram, jitter, duty cycle)
└── power.py             # Power measurements (ripple, PDN impedance, efficiency)

tests/
└── test_measurements.py # 43 unit tests for all measurement modules
```
