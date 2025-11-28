# SpiceLab Troubleshooting Guide

This guide helps you diagnose and fix common issues when working with SpiceLab circuit simulations.

## Quick Diagnostics

SpiceLab includes built-in diagnostic tools:

```python
from spicelab.troubleshooting import Troubleshooter, diagnose_circuit

# Auto-diagnose circuit issues
result = diagnose_circuit(circuit)
print(result)

# Interactive troubleshooting
ts = Troubleshooter(circuit)
ts.diagnose()           # Rich formatted output
ts.interactive()        # Guided Q&A session
ts.quick_fix()          # Just show suggestions
```

### CLI Tools

```bash
# Validate circuit before simulation
spicelab-validate my_circuit.py

# Troubleshoot with detailed diagnostics
spicelab-troubleshoot my_circuit.py --convergence

# Compare netlists
spicelab-diff circuit_v1.py circuit_v2.py
```

---

## Common Errors and Solutions

### E1001: Floating Node

**Symptom:** "Floating nodes detected: n1, n2"

**Cause:** A node is connected to only one component terminal.

**Solutions:**

1. **Add missing connections:**
```python
# Wrong - node 'mid' only connected to R1
circuit.connect(r1.ports[1], Net("mid"))

# Fixed - connect both R1 and R2 to 'mid'
circuit.connect(r1.ports[1], Net("mid"))
circuit.connect(r2.ports[0], Net("mid"))
```

2. **Add a ground reference if intentionally floating:**
```python
# Add high-value resistor to ground
r_leak = Resistor("Rleak", "1G")
circuit.add(r_leak)
circuit.connect(r_leak.ports[0], Net("floating_node"))
circuit.connect(r_leak.ports[1], GND)
```

---

### E1002: Short Circuit

**Symptom:** "Short circuit detected involving: V1, V2"

**Cause:** Two voltage sources connected in parallel (same nodes).

**Solutions:**

1. **Remove duplicate voltage source:**
```python
# Wrong - two voltage sources on same nodes
v1 = Vdc("V1", 5)
v2 = Vdc("V2", 5)  # Remove this
```

2. **Add series resistance:**
```python
# Add small resistance between sources
r_series = Resistor("Rseries", "1m")
circuit.add(r_series)
circuit.connect(v1.ports[0], Net("node_a"))
circuit.connect(r_series.ports[0], Net("node_a"))
circuit.connect(r_series.ports[1], Net("node_b"))
circuit.connect(v2.ports[0], Net("node_b"))
```

---

### E1005: Missing Ground

**Symptom:** "Circuit has no ground reference"

**Cause:** No component connected to the global ground net.

**Solution:**
```python
from spicelab.core.net import GND

# Connect at least one component to GND
circuit.connect(v1.ports[1], GND)
```

---

### E2002: Convergence Failure

**Symptom:** "Simulation failed to converge"

**Causes:**
- Unrealistic component values
- Missing DC path to ground
- Positive feedback without limiting
- Discontinuities in behavioral models

**Solutions:**

1. **Relax convergence tolerance:**
```python
circuit.add_directive(".options reltol=0.01")
circuit.add_directive(".options abstol=1e-10")
```

2. **Add initial conditions:**
```python
circuit.add_directive(".ic V(out)=0")
# or
circuit.add_directive(".nodeset V(bias)=2.5")
```

3. **Ensure DC path to ground for all capacitors:**
```python
# Add high-value parallel resistor
r_bias = Resistor("Rbias", "1G")
circuit.add(r_bias)
circuit.connect(r_bias.ports[0], Net("cap_node"))
circuit.connect(r_bias.ports[1], GND)
```

4. **Use GEAR integration for stiff circuits:**
```python
circuit.add_directive(".options method=gear")
```

5. **Increase iteration limits:**
```python
circuit.add_directive(".options itl1=500")  # DC iterations
circuit.add_directive(".options itl4=100")  # Transient iterations
```

6. **Add GMIN for numerical stability:**
```python
circuit.add_directive(".options gmin=1e-12")
```

---

### E2001: Engine Not Found

**Symptom:** "Engine 'ngspice' not found"

**Cause:** SPICE engine not installed or not on PATH.

**Solutions:**

1. **Install the engine:**
```bash
# macOS
brew install ngspice

# Ubuntu/Debian
sudo apt-get install ngspice

# Windows (via Chocolatey)
choco install ngspice
```

2. **Set environment variable:**
```bash
export SPICELAB_NGSPICE=/path/to/ngspice
```

3. **Run diagnostics:**
```bash
spicelab doctor
```

---

### E3001: Netlist Parse Error

**Symptom:** "Netlist parse error at line X"

**Common causes:**
- Invalid component values
- Syntax errors in directives
- Missing required parameters

**Solutions:**

1. **Check component value syntax:**
```python
# Wrong
r = Resistor("R1", "1K")   # Capital K not standard

# Correct
r = Resistor("R1", "1k")   # Lowercase k
r = Resistor("R1", 1e3)    # Numeric
r = Resistor("R1", "1000") # String numeric
```

2. **Validate SI prefixes:**
```
Valid prefixes:
f = 1e-15 (femto)
p = 1e-12 (pico)
n = 1e-9  (nano)
u = 1e-6  (micro)
m = 1e-3  (milli)
k = 1e3   (kilo)
M = 1e6   (mega, also 'Meg')
G = 1e9   (giga)
T = 1e12  (tera)
```

---

### E4004: Value Out of Range

**Symptom:** Warning about extreme component values

**Common issues:**
- Resistance too high (>1TΩ) or too low (<1mΩ)
- Capacitance too small (<1fF)
- Unrealistic voltage/current sources

**Solutions:**

1. **Check for unit prefix errors:**
```python
# Wrong - 1 millifarad is huge!
c = Capacitor("C1", "1m")

# Correct - 1 microfarad
c = Capacitor("C1", "1u")
```

2. **Use validation to catch issues:**
```python
from spicelab.validators import validate_circuit

result = validate_circuit(circuit)
for warning in result.warnings:
    print(f"{warning.severity}: {warning.message}")
    print(f"  Suggestion: {warning.suggestion}")
```

---

## Empty Results

**Symptom:** Simulation returns no data or all zeros.

**Diagnostic:**
```python
from spicelab.troubleshooting import diagnose_empty_results

result = diagnose_empty_results(circuit, job_result)
print(result)
```

**Common causes and fixes:**

1. **Missing probes:**
```python
# Add voltage probe
circuit.add_directive(".probe v(out)")

# Or probe all nodes
circuit.add_directive(".probe v(*)")
```

2. **Analysis not capturing desired time range:**
```python
# Extend simulation time
# Wrong - signal settles at 2ms, simulation only 1ms
circuit.add_directive(".tran 1u 1m")

# Correct
circuit.add_directive(".tran 1u 10m")
```

3. **Wrong analysis type:**
```python
# For frequency response, use AC not TRAN
circuit.add_directive(".ac dec 100 1 1G")

# For time-domain, use TRAN
circuit.add_directive(".tran 1u 10m")
```

---

## Performance Issues

### Slow Simulation

**Solutions:**

1. **Increase timestep (reduces accuracy):**
```python
# Larger minimum timestep
circuit.add_directive(".tran 10u 10m")  # Instead of 1u
```

2. **Reduce number of data points:**
```python
# Save fewer points
circuit.add_directive(".options numdgt=4")
```

3. **Use behavioral models:**
```python
from spicelab.library import create_component

# Use ideal switch instead of MOSFET for control logic
sw = create_component("behavioral.sw_ideal", "S1")
```

4. **Enable caching:**
```python
from spicelab.orchestrator import run_simulation

result = run_simulation(job, cache=True)
```

### Memory Issues with Monte Carlo

**Solutions:**

1. **Run in batches:**
```python
from spicelab.analysis import monte_carlo

# Instead of 1000 runs at once
for batch in range(10):
    result = monte_carlo(
        circuit, analyses, variations,
        n_runs=100, seed=batch
    )
    process_batch(result)
```

2. **Reduce output data:**
```python
# Only probe nodes of interest
circuit.add_directive(".probe v(out)")  # Not v(*)
```

---

## Validation Before Simulation

Always validate circuits before simulation:

```python
from spicelab.validators import validate_circuit

result = validate_circuit(circuit, strict=False)

if not result.is_valid:
    print("Errors:")
    for error in result.errors:
        print(f"  {error}")

if result.warnings:
    print("Warnings:")
    for warning in result.warnings:
        print(f"  [{warning.severity}] {warning.message}")
        if warning.suggestion:
            print(f"    Suggestion: {warning.suggestion}")
```

### Strict Mode

For production use, enable strict mode to treat warnings as errors:

```python
result = validate_circuit(circuit, strict=True)
# Warnings now appear in result.errors
```

---

## Error Codes Reference

| Code | Category | Description |
|------|----------|-------------|
| E1001 | Circuit | Floating node |
| E1002 | Circuit | Short circuit |
| E1003 | Circuit | Invalid connection |
| E1004 | Circuit | Component not found |
| E1005 | Circuit | Missing ground |
| E2001 | Simulation | Engine not found |
| E2002 | Simulation | Convergence failure |
| E2003 | Simulation | Simulation failed |
| E2004 | Simulation | Analysis error |
| E3001 | Parse | Netlist parse error |
| E3002 | Parse | Result parse error |
| E3003 | Parse | Model parse error |
| E4001 | Validation | Component validation |
| E4002 | Validation | Parameter validation |
| E4003 | Validation | Circuit validation |
| E4004 | Validation | Value out of range |
| E5001 | Config | Engine configuration |
| E5002 | Config | Path not found |

---

## Getting Detailed Help

For any error, get detailed help:

```python
from spicelab.exceptions import ConvergenceError

try:
    result = run_simulation(job)
except ConvergenceError as e:
    print(e)              # Basic message
    print(e.full_help())  # Detailed causes and solutions
```

---

## CLI Diagnostics

### Check Installation

```bash
spicelab doctor
```

This checks:
- Python version
- Required dependencies
- SPICE engines (ngspice, ltspice, xyce)
- Environment variables

### Validate Circuit File

```bash
# Basic validation
spicelab-validate circuit.py

# Strict mode (warnings as errors)
spicelab-validate circuit.py --strict

# JSON output for CI/CD
spicelab-validate circuit.py --json
```

### Interactive Troubleshooting

```bash
spicelab-troubleshoot circuit.py --interactive
```

This guides you through:
1. Identifying the problem type
2. Common causes
3. Step-by-step solutions

---

## Best Practices

1. **Validate early:** Run `validate_circuit()` before simulation
2. **Use templates:** Start with tested templates instead of from scratch
3. **Check units:** Double-check SI prefixes (u vs m vs n)
4. **Add diagnostics:** Use `.probe v(*)` during debugging
5. **Simplify first:** Test with ideal components before using real models
6. **Check ground:** Ensure every circuit has a ground reference
7. **Use caching:** Enable caching for iterative development

---

**Last Updated:** 2025-11-27

**See Also:**
- [Component Catalog](component_catalog.md) - Available components
- [Template Catalog](template_catalog.md) - Pre-built circuit templates
- [IDE Setup Guide](ide_setup.md) - Editor integration
