# Implementation Tasks - M8 Model Management

## 1. Model Registry Infrastructure
- [ ] 1.1 Create `spicelab/models/registry.py` with ModelRegistry class
- [ ] 1.2 Implement ModelRef Pydantic model (kind, path, name, params)
- [ ] 1.3 Add model search path resolution (project dir, user dir, system)
- [ ] 1.4 Implement model version tracking and validation
- [ ] 1.5 Add model caching mechanism (hash-based)

## 2. Model Types Support
- [ ] 2.1 Support SPICE `.model` references
- [ ] 2.2 Support subcircuit `.subckt` references
- [ ] 2.3 Support library `.lib` file inclusion
- [ ] 2.4 Support Verilog-A module references
- [ ] 2.5 Add model parameter override mechanism

## 3. Verilog-A Pipeline
- [ ] 3.1 Create `spicelab/models/veriloga.py`
- [ ] 3.2 Implement ADMS toolchain detection (where/which)
- [ ] 3.3 Add Verilog-A compilation wrapper for Xyce
- [ ] 3.4 Add Verilog-A to ngspice compatibility checking
- [ ] 3.5 Implement error message mapping (parse ADMS errors)
- [ ] 3.6 Add compiled model caching by source file hash

## 4. Netlist Writer Integration
- [ ] 4.1 Extend Circuit to accept ModelRef instances
- [ ] 4.2 Update ngspice writer to emit `.include` and `.lib` directives
- [ ] 4.3 Update LTspice writer to handle .lib syntax differences
- [ ] 4.4 Update Xyce writer for Verilog-A model references
- [ ] 4.5 Add model dependency ordering (includes before usage)

## 5. Error Handling and Validation
- [ ] 5.1 Validate model file existence before simulation
- [ ] 5.2 Check engine feature support (Verilog-A, .lib syntax)
- [ ] 5.3 Provide actionable errors: "Model 'opamp741' not found in paths: [...]"
- [ ] 5.4 Detect circular .include dependencies
- [ ] 5.5 Warn when engine doesn't support requested model type

## 6. Model Library Utilities
- [ ] 6.1 Implement model download helper (optional, for common libs)
- [ ] 6.2 Add model introspection (list params, pins from .subckt)
- [ ] 6.3 Create model validation (syntax check before simulation)
- [ ] 6.4 Add model documentation extractor (comments from .lib files)

## 7. Testing and Examples
- [ ] 7.1 Create test with custom `.subckt` (opamp macro-model)
- [ ] 7.2 Create test with SPICE `.model` (diode, MOSFET)
- [ ] 7.3 Create test with Verilog-A model (if ADMS available)
- [ ] 7.4 Test model path resolution fallback chain
- [ ] 7.5 Test error messages for missing models
- [ ] 7.6 Create example: "Using standard opamp from library"
- [ ] 7.7 Create example: "Custom Verilog-A model in Xyce"

## 8. Documentation
- [ ] 8.1 Write `docs/models.md` with model management guide
- [ ] 8.2 Document Verilog-A setup (ADMS installation per OS)
- [ ] 8.3 Document model search path configuration
- [ ] 8.4 Add model library structure best practices
- [ ] 8.5 Document engine-specific model format differences
