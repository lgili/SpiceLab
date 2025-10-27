# M8 - Model Management and Verilog-A Support

## Why
SpiceLab needs robust support for device models, subcircuits, and Verilog-A behavioral models. Currently, there's no standardized way to manage model libraries, resolve paths, or compile Verilog-A models for different engines. This limits usage with real PDKs and custom device models critical for professional analog design.

## What Changes
- Add `spicelab/models/registry.py` for centralized model management with versioning
- Implement `ModelRef` type for referencing models (subcircuit, SPICE model, Verilog-A)
- Add `spicelab/models/veriloga.py` with ADMS pipeline helpers for Verilog-A compilation
- Extend netlist writers to support `.include`, `.lib`, and Verilog-A references per engine
- Add model path resolution with search paths and caching
- Provide clear error messages when models are missing or engine doesn't support a feature

## Impact
- **Affected specs**: models, verilog-a, netlist-generation
- **Affected code**:
  - New: `spicelab/models/registry.py`, `spicelab/models/veriloga.py`
  - Modified: `spicelab/engines/ngspice_proc.py`, `ltspice_cli.py`, `xyce_cli.py` (netlist writers)
  - Modified: `spicelab/core/types.py` (add ModelRef support to Circuit/Component)
  - New: `tests/test_model_registry.py`, `examples/models/`
- **Dependencies**: None (Verilog-A compilation requires external ADMS toolchain)
