# M3: Component Library Expansion - Tasks

**Status:** Proposed
**Start Date:** TBD
**Target Completion:** TBD (6-8 weeks)
**Dependencies:** M1 (validation framework), M2 (performance optimization)

## Task Breakdown

### Phase 1: Infrastructure (Week 1)
- [ ] Create package structure
  - [ ] Create `spicelab/library/` package
  - [ ] Add subdirectories: `passive/`, `active/`, `templates/`, `behavioral/`
  - [ ] Create `spicelab/models/` package
  - [ ] Add subdirectories: `catalog/`, `database.py`, `downloader.py`, `validator.py`
- [ ] Setup model database
  - [ ] Design SQLite schema (models table)
  - [ ] Implement ModelDatabase class
  - [ ] Add methods: add_model(), search(), get()
  - [ ] Create database migration system
  - [ ] Add database initialization (~/.spicelab/models.db)
- [ ] Implement model downloader
  - [ ] Create ModelDownloader class
  - [ ] Add caching (~/.spicelab/cache/)
  - [ ] Implement checksum verification (SHA256)
  - [ ] Add progress reporting for downloads
  - [ ] Handle network errors gracefully
- [ ] Create model validator
  - [ ] Implement ModelValidator class
  - [ ] Add SPICE syntax checking
  - [ ] Validate .MODEL and .SUBCKT
  - [ ] Check for unmatched .ENDS
  - [ ] Validate parameter syntax
  - [ ] Add tests for validator

**Estimated Time:** 1 week
**Assignee:** TBD

---

### Phase 2: Basic Component Library (Week 2)
- [ ] E-series resistors
  - [ ] Define E12, E24, E96 value arrays
  - [ ] Implement ESeriesResistor class
  - [ ] Add E12(), E24(), E96() factory methods
  - [ ] Implement nearest_E12(), nearest_E24() helpers
  - [ ] Add convenience functions: R_E12(), R_E24()
  - [ ] Write tests (all E-series values)
  - [ ] Document with examples
- [ ] Ceramic capacitor catalog
  - [ ] Define CapacitorSpec dataclass
  - [ ] Create CERAMIC_CAP_CATALOG dictionary (100+ values)
  - [ ] Implement CeramicCapacitor class
  - [ ] Add dielectric types (C0G, X7R, X5R, Y5V)
  - [ ] Add voltage ratings (6.3V, 16V, 25V, 50V)
  - [ ] Include common SMD packages (0402, 0603, 0805, 1206)
  - [ ] Write tests
  - [ ] Document catalog
- [ ] Inductor catalog
  - [ ] Create standard inductor values (1uH - 1mH)
  - [ ] Add saturation current ratings
  - [ ] Include DCR (DC resistance) specs
  - [ ] Add common packages (0603, 0805, radial)
  - [ ] Write tests
- [ ] Diode catalog
  - [ ] Add 1N4148 (switching diode)
  - [ ] Add 1N4007 (rectifier)
  - [ ] Add 1N5819 (Schottky)
  - [ ] Add Zener diodes (3.3V, 5.1V, 12V)
  - [ ] Include reverse voltage and current ratings
- [ ] BJT catalog
  - [ ] Add 2N2222 (NPN, general purpose)
  - [ ] Add 2N3904 (NPN, small signal)
  - [ ] Add 2N3906 (PNP, small signal)
  - [ ] Add BC547/BC557 (European equivalents)
  - [ ] Include beta, Vce(sat), and Ic(max) specs
- [ ] MOSFET catalog
  - [ ] Add 2N7000 (N-channel, small signal)
  - [ ] Add IRF540 (N-channel, power)
  - [ ] Add BSS138 (N-channel, logic level)
  - [ ] Add IRF9540 (P-channel, power)
  - [ ] Include Vgs(th), Rds(on), Id(max) specs

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
