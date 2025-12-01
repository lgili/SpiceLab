# SpiceLab Tutorial Series

Welcome to the SpiceLab tutorial series! These tutorials will guide you from basic concepts to advanced circuit simulation techniques.

## Getting Started

1. **[Chapter 1: Getting Started](01_getting_started.md)** - Installation and your first circuit
   - Installing SpiceLab and SPICE engines
   - Building a voltage divider
   - Understanding circuits, components, and nets
   - Circuit validation

2. **[Chapter 2: Basic Components](02_basic_components.md)** - Passive and active components
   - Resistors, capacitors, inductors
   - DC, AC, pulse, and sine voltage sources
   - Current sources
   - Component value notation

3. **[Chapter 3: Building Circuits](03_building_circuits.md)** - Advanced construction
   - Programmatic circuit building
   - Factory functions and templates
   - Combining circuits
   - Adding SPICE directives

## Simulation

4. **[Chapter 4: Analysis Types](04_analysis_types.md)** - Running simulations
   - DC operating point (.op)
   - DC sweep (.dc)
   - AC frequency analysis (.ac)
   - Transient analysis (.tran)

5. **[Chapter 5: Working with Results](05_results.md)** - Processing data
   - Understanding xarray datasets
   - Extracting measurements
   - Plotting with matplotlib
   - Exporting data

## Advanced Topics

6. **[Chapter 6: Monte Carlo Analysis](06_monte_carlo.md)** - Statistical simulations
   - Tolerance distributions (NormalPct, UniformPct)
   - Running Monte Carlo simulations
   - Yield analysis and sensitivity
   - Histogram visualization

7. **[Chapter 7: Advanced Components](07_advanced_components.md)** - Real-world parts
   - Vendor models (diodes, transistors, op-amps)
   - Custom subcircuits
   - Behavioral models
   - E-series component selection

8. **[Chapter 8: Performance](08_performance.md)** - Optimization
   - Parallel execution with workers
   - Result caching
   - Memory management
   - Profiling simulations

9. **[Chapter 9: Integration](09_integration.md)** - Automation
   - CI/CD with GitHub Actions
   - Batch processing
   - Jupyter notebooks
   - Report generation

10. **[Chapter 10: Troubleshooting](10_troubleshooting.md)** - Debugging
    - Convergence failures
    - Validation errors
    - Diagnostic tools
    - Best practices

## Prerequisites

- Python 3.10 or later
- Basic electronics knowledge (Ohm's law, filters, etc.)
- A SPICE engine (ngspice recommended)

## Running Tutorial Examples

All tutorials have corresponding example files:

```bash
# Run from repository root
python examples/tutorials/01_first_circuit.py
python examples/tutorials/02_rc_filter.py
# ... etc.
```

## Quick Reference

| Topic | Key Imports |
|-------|-------------|
| Circuit basics | `from spicelab.core.circuit import Circuit` |
| Components | `from spicelab.core.components import Resistor, Capacitor, Vdc, ...` |
| Nets | `from spicelab.core.net import GND, Net` |
| Analysis | `from spicelab.core.types import AnalysisSpec` |
| Simulation | `from spicelab.engines import run_simulation` |
| Templates | `from spicelab.templates import rc_lowpass, butterworth_lowpass, ...` |
| Monte Carlo | `from spicelab.analysis import monte_carlo, NormalPct` |

## Getting Help

- **Documentation**: [SpiceLab Docs](../index.md)
- **Component Reference**: [Component Catalog](../component_catalog.md)
- **Templates**: [Template Catalog](../template_catalog.md)
- **Troubleshooting**: [Troubleshooting Guide](../troubleshooting_guide.md)
- **IDE Setup**: [IDE Setup Guide](../ide_setup.md)

---

**Last Updated:** 2025-11-28
