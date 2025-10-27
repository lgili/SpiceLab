# Usability Improvements (Quality of Life Features)

## Why
Based on analysis of the roadmap, there are several usability improvements that would make SpiceLab significantly easier to use for both beginners and advanced users. These are "quality of life" features that reduce friction, prevent common mistakes, and accelerate workflows.

## What Changes
Add cross-cutting usability improvements across the entire library:

**Smart Defaults and Presets**
- Common circuit templates (RC filter, opamp, ADC stages)
- Analysis presets (quick AC, detailed transient, noise analysis)
- Default parameter ranges based on component type
- Auto-detection of analysis type from circuit structure

**Improved Error Prevention**
- Parameter validation with helpful suggestions (e.g., "C1=10u → did you mean 10µF?")
- Unit parsing with automatic conversion (1k, 1kohm, 1000 all work)
- Connectivity checking before simulation (detect floating nodes, shorts)
- Component value sanity checks (warn if R=1e-6 Ω or C=1 F)

**Workflow Shortcuts**
- Quick simulation: `circuit.quick_ac()`, `circuit.quick_tran()`
- Chaining: `circuit.simulate().measure().plot()`
- Automatic plot selection based on analysis type
- Common measurement shortcuts: `.pm()`, `.bw()`, `.overshoot()`

**Better Debugging**
- Verbose mode with step-by-step logging
- Dry-run mode (validate without simulating)
- Netlist preview before simulation
- Interactive mode for ambiguous situations (ask user instead of failing)

**Documentation Helpers**
- Inline examples in docstrings (runnable)
- `circuit.help()` context-sensitive documentation
- Tutorial mode with guided workflows
- Auto-completion hints for IDE integration

## Impact
- **Affected specs**: usability, defaults, error-handling, shortcuts, documentation
- **Affected code**:
  - Modified: All public APIs to add validation and helpful defaults
  - New: `spicelab/templates/` for circuit and analysis templates
  - New: `spicelab/shortcuts/` for workflow shortcuts
  - Modified: Error messages across entire codebase
  - Enhanced: All docstrings with examples
- **Dependencies**: None (pure Python improvements)
