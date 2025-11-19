# Implementation Tasks - Usability Improvements

## 1. Smart Defaults and Presets
- [x] 1.1 Create `spicelab/templates/circuits.py` with common circuits
- [x] 1.2 Add RC filter template with customizable fc
- [x] 1.3 Add voltage divider template
- [ ] 1.4 Add opamp non-inverting/inverting amplifier templates
- [ ] 1.5 Create analysis presets (quick_ac, detailed_tran, etc.)
- [ ] 1.6 Auto-detect suitable analysis from circuit topology

## 2. Parameter Validation and Unit Handling
- [x] 2.1 Implement flexible unit parser: 1k, 1kohm, 1e3, 1000 all valid
- [x] 2.2 Add "did you mean" suggestions for common typos (10u → 10µ)
- [x] 2.3 Validate parameter ranges and warn on unusual values
- [x] 2.4 Add unit system with automatic SI prefix handling
- [x] 2.5 Support engineering notation: 1.5M, 22p, 470n

## 3. Circuit Validation
- [ ] 3.1 Detect floating nodes before simulation
- [ ] 3.2 Detect short circuits (voltage sources in parallel)
- [ ] 3.3 Detect open circuits (current sources in series)
- [x] 3.4 Warn on missing ground reference
- [x] 3.5 Check for valid component connections (2-pin, 3-pin, etc.)

## 4. Workflow Shortcuts
- [x] 4.1 Add `quick_ac(start=1, stop=1e9)` with smart defaults
- [x] 4.2 Add `quick_tran(duration="1ms")` with auto-timestep
- [ ] 4.3 Implement method chaining: `circuit.simulate().measure().plot()`
- [ ] 4.4 Add common measurement shortcuts on Result: `.pm()`, `.bw()`, `.overshoot()`
- [ ] 4.5 Add auto-plot selection based on analysis type

## 5. Error Handling Improvements
- [ ] 5.1 Add context to all exceptions (what was being attempted)
- [ ] 5.2 Provide recovery suggestions in error messages
- [ ] 5.3 Add error codes for programmatic handling
- [ ] 5.4 Create error message catalog with solutions
- [ ] 5.5 Implement "did you mean?" for method/attribute typos

## 6. Interactive and Debugging Features
- [ ] 6.1 Add verbose mode: `sim.run(..., verbose=True)` with progress logging
- [ ] 6.2 Implement dry-run: `sim.run(..., dry_run=True)` validates without simulating
- [ ] 6.3 Add netlist preview: `circuit.preview_netlist(engine="ngspice")`
- [ ] 6.4 Add interactive mode for ambiguous choices (prompt user)
- [ ] 6.5 Implement simulation step-by-step debugger

## 7. Documentation and Help
- [ ] 7.1 Add runnable examples to all docstrings
- [ ] 7.2 Implement context-sensitive help: `circuit.help()`, `Result.help()`
- [ ] 7.3 Create interactive tutorial mode: `spicelab tutorial`
- [ ] 7.4 Add "see also" links in docstrings
- [ ] 7.5 Generate API cheat sheet (PDF/HTML quick reference)

## 8. Type Hints and IDE Support
- [ ] 8.1 Add comprehensive type hints to all public APIs
- [ ] 8.2 Create .pyi stub files for better IDE completion
- [ ] 8.3 Add type validation decorators
- [ ] 8.4 Test type checking with mypy in strict mode
- [ ] 8.5 Add docstring examples that are type-checkable

## 9. Common Patterns Library
- [ ] 9.1 Create `patterns/` with reusable circuit patterns
- [ ] 9.2 Add pattern: frequency compensation networks
- [ ] 9.3 Add pattern: bias networks (current mirrors, bandgaps)
- [ ] 9.4 Add pattern: filter topologies (Butterworth, Chebyshev)
- [ ] 9.5 Add pattern: ADC/DAC building blocks

## 10. User Experience Polish
- [ ] 10.1 Add progress bars for long operations (with ETA)
- [ ] 10.2 Implement undo/redo for circuit modifications
- [ ] 10.3 Add clipboard support (copy/paste circuit snippets)
- [ ] 10.4 Create circuit diff tool (visualize changes)
- [ ] 10.5 Add bookmark/favorites for frequently used configurations

## 11. Testing
- [x] 11.1 Test all templates produce valid circuits
- [ ] 11.2 Test error message clarity with user studies
- [x] 11.3 Test shortcuts cover 80% of common workflows
- [ ] 11.4 Validate documentation examples are runnable
- [ ] 11.5 Test type hints with mypy and pyright

## 12. Documentation
- [ ] 12.1 Write `docs/usability-guide.md` with best practices
- [ ] 12.2 Create "Common Pitfalls" guide
- [ ] 12.3 Document all templates and presets
- [ ] 12.4 Create video tutorials for key workflows
