# M3: Component Library Expansion - Tasks

**Status:** In Progress
**Start Date:** 2025-11-24
**Target Completion:** TBD (6-8 weeks)
**Dependencies:** M1 (validation framework), M2 (performance optimization)

## Task Breakdown

### Phase 1: Infrastructure (Week 1) - PARTIALLY COMPLETE (Pre-existing)
- [x] Create package structure
  - [x] `spicelab/library/` package exists
  - [x] Registry system with `register_component()`, `list_components()`, `create_component()`
  - [x] Factory typing protocols for components
  - [x] Metadata support for component specs
  - [x] `spicelab/library/behavioral.py` added for behavioral models
- [ ] Setup model database (Future - external vendor models)
- [ ] Implement model downloader (Future - external vendor models)
- [ ] Create model validator (Future - external vendor models)

**Completed:** 2025-11-24 (using existing infrastructure)

---

### Phase 2: Basic Component Library (Week 2) ✅ COMPLETE
- [x] E-series resistors
  - [x] E12, E24, E48, E96, E192 in `spicelab/utils/e_series.py`
  - [x] `round_to_series()`, `nearest_value()`, `find_best_match()`
  - [x] Series/parallel combination finding
- [x] Diode catalog - 18 diodes total
  - [x] Signal: 1N4148, 1N914
  - [x] Rectifier: 1N4001, 1N4004, 1N4007
  - [x] Schottky: 1N5817, 1N5818, 1N5819, BAT54
  - [x] Zener: 1N4728A (3.3V), 1N4733A (5.1V), 1N4742A (12V), 1N4744A (15V), BZX55C5V1
  - [x] LEDs: Red, Green, Blue, White
- [x] BJT catalog - 15 BJTs total
  - [x] NPN: 2N2222, 2N2222A, 2N3904, 2N4401, BC547B, BC548, MPSA06
  - [x] PNP: 2N2907, 2N3906, 2N4403, BC557, BC558, MPSA56
  - [x] Darlington: TIP120 (NPN), TIP125 (PNP)
- [x] MOSFET catalog - 11 MOSFETs total
  - [x] N-channel small-signal: 2N7000, BSS138, BS170
  - [x] N-channel power: IRF540N, IRFZ44N, IRF3205, IRLZ44N (logic-level)
  - [x] P-channel small-signal: BS250
  - [x] P-channel power: IRF9540N, AO3401A, IRF4905

**Completed:** 2025-11-24

---

### Phase 3: Behavioral Models ✅ COMPLETE
- [x] Ideal diode models
  - [x] D_IDEAL (near-zero Vf)
  - [x] D_IDEAL_SCHOTTKY (0.2V Vf)
  - [x] D_IDEAL_Z5V1 (5.1V Zener)
- [x] Ideal switch models
  - [x] SW_IDEAL (Ron=1mΩ, Roff=1GΩ)
  - [x] SW_IDEAL_FAST (narrow hysteresis)
  - [x] SW_RELAY (relay-like thresholds)
- [x] Transformer info (for coupled inductors)
  - [x] 1:1 isolation
  - [x] 10:1 step-down
  - [x] Flyback 5:1
- [x] Controlled sources (pre-existing in core/components.py)
  - [x] VCVS, VCCS, CCCS, CCVS

**Completed:** 2025-11-24

---

### Phase 4: Tests ✅ COMPLETE
- [x] Library expansion tests
  - [x] 37 tests in `tests/test_library_expanded.py`
  - [x] Tests for diode catalog
  - [x] Tests for BJT catalog
  - [x] Tests for MOSFET catalog
  - [x] Tests for behavioral models
  - [x] Metadata validation tests

**Completed:** 2025-11-24

---

### Phase 5: Vendor Model Research (Week 3) - DEFERRED
- [ ] Research vendor model sources (TI, ADI, Infineon, etc.)
- [ ] Select initial model set (50+ models)
- [ ] Download and organize

**Status:** Deferred to future milestone (requires infrastructure from Phase 1)

---

### Phase 6: Documentation - PENDING
- [ ] Component catalog reference
- [ ] Behavioral model guide
- [ ] API documentation updates

**Status:** Pending

---

## Current Library Summary

**Total Components:** 68

| Category    | Count | Description                                      |
|------------|-------|--------------------------------------------------|
| diode      | 18    | Signal, rectifier, Schottky, Zener, LEDs        |
| bjt        | 15    | NPN/PNP general-purpose, low-noise, Darlington  |
| mosfet     | 11    | N/P-channel, small-signal, power, logic-level   |
| behavioral | 11    | Ideal diodes, switches, transformers            |
| opamp      | 3     | Ideal, LM741, TL081                             |
| capacitor  | 3     | Ceramic, electrolytic, film                     |
| resistor   | 3     | Metal film, carbon film, shunt                  |
| inductor   | 2     | Standard inductors                              |
| switch     | 2     | Voltage/current controlled switches             |

---

### Original Phase 2: Basic Component Library (Week 2) - Reference
- [ ] E-series resistors (existing in e_series.py)
- [ ] Ceramic capacitor catalog (basic entries exist)
- [ ] Inductor catalog (basic entries exist)
- [x] Diode catalog - EXPANDED
- [x] BJT catalog - EXPANDED
- [x] MOSFET catalog - EXPANDED

**Estimated Time:** 1 week
**Assignee:** TBD

---

### Phase 3: Vendor Model Research (Week 3)
- [ ] Research vendor model sources
  - [ ] Survey TI model library (TINA-TI, PSpice)
  - [ ] Survey Analog Devices (LTspice library)
  - [ ] Survey Infineon (Infineon Designer)
  - [ ] Survey STMicroelectronics (ST website)
  - [ ] Survey NXP (NXP models)
  - [ ] Document download URLs and terms
- [ ] Select initial model set (50+ models)
  - [ ] 15 OpAmps (LM358, TL071, OP27, AD8065, OPA2134, etc.)
  - [ ] 10 Voltage Regulators (LM317, LM7805, TPS54331, ADP3050, etc.)
  - [ ] 10 MOSFETs (IRF540, BSS138, IRFZ44N, Si2302, etc.)
  - [ ] 5 Comparators (LM311, LM393, TLV3501, etc.)
  - [ ] 5 ADCs (ADS1115, MCP3008, AD7606, etc.)
  - [ ] 5 DACs (MCP4725, AD5691, DAC8551, etc.)
- [ ] Download and organize
  - [ ] Download all selected models
  - [ ] Organize by vendor/category
  - [ ] Rename files consistently (PartNumber.lib)
  - [ ] Create manifest.json (master index)
  - [ ] Create per-vendor index.json files

**Estimated Time:** 1 week
**Assignee:** TBD

---

### Phase 4: Vendor Model Integration (Week 4)
- [ ] Validate downloaded models
  - [ ] Run ModelValidator on all 50+ models
  - [ ] Fix syntax errors (if any)
  - [ ] Verify .SUBCKT/.ENDS matching
  - [ ] Check for non-ASCII characters
  - [ ] Document any known issues
- [ ] Populate model database
  - [ ] Add all models to SQLite database
  - [ ] Include metadata (vendor, part, category, description)
  - [ ] Add download URLs
  - [ ] Compute and store checksums
  - [ ] Mark as verified
- [ ] Create catalog structure
  - [ ] Setup spicelab/models/catalog/ directory
  - [ ] Organize ti/, adi/, infineon/, stm/, nxp/
  - [ ] Create category subdirectories (opamps/, regulators/, etc.)
  - [ ] Copy model files to catalog
  - [ ] Generate manifest.json
- [ ] Implement catalog loader
  - [ ] Create ModelCatalog class
  - [ ] Implement list_vendors()
  - [ ] Implement list_models(vendor, category)
  - [ ] Implement get_model_path(vendor, part_number)
  - [ ] Add error handling (model not found)
  - [ ] Write tests for catalog loading
- [ ] Test model loading
  - [ ] Test all 50+ models load correctly
  - [ ] Verify SPICE syntax is valid
  - [ ] Test subcircuit instantiation
  - [ ] Measure catalog load time (<100ms target)

**Estimated Time:** 1 week
**Assignee:** TBD

---

### Phase 5: Subcircuit Templates (Week 5)
- [ ] Generic OpAmp template
  - [ ] Implement OpAmpTemplate class
  - [ ] Parameters: gain, bandwidth, slew_rate, input_Z, output_Z
  - [ ] Build behavioral model (VCVS + RC)
  - [ ] Generate .SUBCKT SPICE code
  - [ ] Add ports: in+, in-, out, vcc, vee
  - [ ] Test with non-inverting amplifier circuit
  - [ ] Document usage and limitations
- [ ] LDO regulator template
  - [ ] Implement LDOTemplate class
  - [ ] Parameters: v_out, i_max, dropout, psrr
  - [ ] Build behavioral model (simplified)
  - [ ] Add ports: vin, vout, gnd, enable
  - [ ] Test with load variations
  - [ ] Document usage
- [ ] Buck converter template
  - [ ] Implement BuckConverterTemplate
  - [ ] Parameters: v_in, v_out, frequency, i_max
  - [ ] Add switch, diode, inductor, capacitor
  - [ ] Include PWM control model
  - [ ] Test transient response
  - [ ] Document design equations
- [ ] Boost converter template
  - [ ] Implement BoostConverterTemplate
  - [ ] Parameters: v_in, v_out, frequency, i_max
  - [ ] Add switch, diode, inductor, capacitor
  - [ ] Test with varying loads
  - [ ] Document usage
- [ ] Flyback template (optional)
  - [ ] Implement FlybackTemplate
  - [ ] Add transformer model
  - [ ] Include isolated feedback
  - [ ] Document usage
- [ ] ADC/DAC behavioral models
  - [ ] Implement IdealADC (behavioral)
  - [ ] Implement IdealDAC (behavioral)
  - [ ] Add quantization effects
  - [ ] Test with signal chains

**Estimated Time:** 1 week
**Assignee:** TBD

---

### Phase 6: Behavioral Models (Week 6)
- [ ] Ideal switch
  - [ ] Implement IdealSwitch class
  - [ ] Parameters: r_on, r_off, v_threshold
  - [ ] Generate SPICE .MODEL VSWITCH
  - [ ] Test with PWM circuit
  - [ ] Document usage
- [ ] Ideal diode
  - [ ] Implement IdealDiode class
  - [ ] Parameters: v_forward (near-zero)
  - [ ] Generate SPICE .MODEL D
  - [ ] Test with rectifier circuit
  - [ ] Document limitations
- [ ] Ideal transformer
  - [ ] Implement IdealTransformer class
  - [ ] Parameters: turns_ratio, coupling_factor
  - [ ] Use coupled inductors model (K statement)
  - [ ] Test with AC analysis
  - [ ] Document usage
- [ ] Controlled sources
  - [ ] Implement VCVS (voltage-controlled voltage source)
  - [ ] Implement VCCS (voltage-controlled current source)
  - [ ] Implement CCVS (current-controlled voltage source)
  - [ ] Implement CCCS (current-controlled current source)
  - [ ] Add polynomial and table-based sources
  - [ ] Test with control systems
- [ ] Limiters
  - [ ] Implement VoltageLimiter (clamp to ±Vmax)
  - [ ] Implement CurrentLimiter
  - [ ] Implement SlewRateLimiter
  - [ ] Test with signal processing circuits
  - [ ] Document usage

**Estimated Time:** 1 week
**Assignee:** TBD

---

### Phase 7: Documentation (Week 7)
- [ ] Component catalog reference
  - [ ] Create `docs/component_catalog.md`
  - [ ] List all basic components with specs
  - [ ] Add usage examples for each
  - [ ] Include part selection guides
- [ ] Vendor model usage guide
  - [ ] Create `docs/vendor_models.md`
  - [ ] Document how to search models
  - [ ] Explain model downloading
  - [ ] Show vendor model integration in circuits
  - [ ] Add troubleshooting section
- [ ] Subcircuit template tutorials
  - [ ] Create `docs/subcircuit_templates.md`
  - [ ] Tutorial for each template (OpAmp, LDO, Buck, etc.)
  - [ ] Show complete example circuits
  - [ ] Explain parameter selection
  - [ ] Document limitations
- [ ] Behavioral model guide
  - [ ] Create `docs/behavioral_models.md`
  - [ ] Document all behavioral models
  - [ ] Explain use cases (system-level simulation)
  - [ ] Show examples (PWM, rectifier, etc.)
  - [ ] Document accuracy limitations
- [ ] API documentation
  - [ ] Add docstrings to all new classes
  - [ ] Generate API reference (Sphinx/MkDocs)
  - [ ] Add code examples in docstrings
  - [ ] Cross-reference related components

**Estimated Time:** 1 week
**Assignee:** TBD

---

### Phase 8: Integration & Testing (Week 8)
- [ ] Write unit tests
  - [ ] Test ESeriesResistor (all values valid)
  - [ ] Test CeramicCapacitor catalog
  - [ ] Test ModelDatabase (CRUD operations)
  - [ ] Test ModelDownloader (caching, checksums)
  - [ ] Test ModelValidator (valid/invalid models)
  - [ ] Test ModelCatalog (loading, searching)
  - [ ] Test all subcircuit templates
  - [ ] Test all behavioral models
- [ ] Integration tests
  - [ ] Build circuit with E-series resistors
  - [ ] Load and simulate vendor OpAmp model
  - [ ] Use LDO template in power supply circuit
  - [ ] Combine behavioral models in system
  - [ ] Test catalog load performance (<100ms)
- [ ] Example circuits
  - [ ] Create 10+ example circuits using new components
  - [ ] Examples: non-inverting amp, buck converter, ADC signal chain
  - [ ] Add to examples/ directory
  - [ ] Include README with explanation
- [ ] Performance validation
  - [ ] Measure catalog load time (target: <100ms)
  - [ ] Measure model search time (target: <10ms)
  - [ ] Test memory usage (large catalog)
  - [ ] Ensure no performance regressions
- [ ] User acceptance testing
  - [ ] Get feedback from beta users
  - [ ] Address usability issues
  - [ ] Refine API based on feedback

**Estimated Time:** 1 week
**Assignee:** TBD

---

## Acceptance Criteria

### Must Have
- [ ] **100+** basic components (passive + active)
- [ ] **50+** vendor models integrated and validated
- [ ] **10+** subcircuit templates (OpAmp, LDO, Buck, Boost, etc.)
- [ ] **10+** behavioral models (switches, sources, limiters)
- [ ] All models pass SPICE syntax validation
- [ ] All components documented with examples
- [ ] Comprehensive unit test coverage (>95%)
- [ ] Catalog loads in <100ms

### Should Have
- [ ] Model downloader with caching
- [ ] Searchable model database (vendor, category, part)
- [ ] Example circuits demonstrating new components
- [ ] Performance benchmarks (no regressions)
- [ ] User guide for vendor models

### Nice to Have
- [ ] Web scraper for automated vendor model updates
- [ ] Component recommendation system
- [ ] Interactive component browser (CLI)
- [ ] VSCode autocomplete for component catalog

## Testing Checklist

Before marking M3 as complete:
- [ ] All 100+ basic components load correctly
- [ ] All 50+ vendor models validate successfully
- [ ] Model database queries return correct results
- [ ] Downloader caches files properly (no duplicate downloads)
- [ ] All subcircuit templates generate valid SPICE
- [ ] All behavioral models simulate correctly
- [ ] Catalog load time <100ms (tested on CI)
- [ ] All unit tests pass (>95% coverage)
- [ ] Integration tests pass (example circuits simulate)
- [ ] Documentation complete and accurate

## Component Inventory

### Basic Components (100+)
- **Resistors:** E12 (12) + E24 (24) + E96 (96) = 132 values
- **Capacitors:** 100+ ceramic (various dielectrics/voltages)
- **Inductors:** 20+ standard values
- **Diodes:** 10+ (signal, rectifier, Schottky, Zener)
- **BJTs:** 10+ (NPN/PNP, signal/power)
- **MOSFETs:** 10+ (N/P channel, logic/power)

### Vendor Models (50+)
- **TI:** 20+ (OpAmps, regulators, comparators)
- **Analog Devices:** 15+ (OpAmps, ADCs, DACs)
- **Infineon:** 10+ (MOSFETs, IGBTs)
- **STMicroelectronics:** 5+ (regulators, drivers)
- **NXP:** 5+ (logic, MOSFETs)

### Templates (10+)
- Generic OpAmp
- LDO regulator
- Buck converter
- Boost converter
- Flyback transformer
- ADC behavioral
- DAC behavioral
- Comparator
- PWM controller
- PLL (optional)

### Behavioral Models (10+)
- Ideal switch (voltage-controlled)
- Ideal diode
- Ideal transformer
- VCVS, VCCS, CCVS, CCCS
- Voltage limiter
- Current limiter
- Slew rate limiter

## Notes

- **Licensing:** Only include models with permissive licenses or provide downloader
- **File Size:** Keep bundled catalog <10MB, download large models on-demand
- **Accuracy:** All models reference vendor datasheets, include disclaimers
- **Compatibility:** Test models with NGSpice, LTspice, Xyce
- **Versioning:** Track model versions in database for updates

## Dependencies

- M1 (Stability) - validation framework for model checking
- M2 (Performance) - catalog must load quickly for large libraries

## Blocking

- M4 (DX) - autocomplete depends on component catalog
- M5 (Docs) - examples depend on rich component library
- M13 (Mixed-Signal) - ADC/DAC models are foundation

---

**Last Updated:** 2025-01-19
