# Chapter 10: Troubleshooting

This chapter covers diagnosing and fixing common simulation problems: convergence failures, empty results, and validation errors.

## Diagnostic Tools

### Automatic Diagnosis

```python
from spicelab.troubleshooting import Troubleshooter, diagnose_circuit

# Quick diagnosis
result = diagnose_circuit(circuit)
print(result)

# Interactive troubleshooting
ts = Troubleshooter(circuit)
ts.diagnose()           # Rich formatted output
ts.interactive()        # Guided Q&A session
ts.quick_fix()          # Just suggestions
```

### CLI Diagnostics

```bash
# Validate circuit
spicelab-validate circuit.py

# Troubleshoot with details
spicelab-troubleshoot circuit.py --convergence

# Check environment
spicelab doctor
```

## Common Problems and Solutions

### 1. Floating Node

**Error:** `Floating nodes detected: n1, n2`

**Cause:** A node is connected to only one component terminal.

**Solutions:**

```python
# Option 1: Add missing connection
circuit.connect(r2.ports[0], Net("n1"))  # Connect second component

# Option 2: Add high-value resistor to ground (if intentionally floating)
r_leak = Resistor("LEAK", "1G")
circuit.add(r_leak)
circuit.connect(r_leak.ports[0], Net("floating_node"))
circuit.connect(r_leak.ports[1], GND)
```

### 2. Missing Ground

**Error:** `Circuit has no ground reference`

**Cause:** No component is connected to GND.

**Solution:**

```python
from spicelab.core.net import GND

# Connect at least one component to GND
circuit.connect(v1.ports[1], GND)
```

### 3. Voltage Source Loop

**Error:** `Voltage source loop detected: V1, V2`

**Cause:** Two voltage sources connected in parallel (same nodes).

**Solutions:**

```python
# Option 1: Remove duplicate source
circuit.remove(v2)

# Option 2: Add series resistance
r_series = Resistor("SER", "1m")  # 1 milliohm
circuit.add(r_series)
# Insert between sources
```

### 4. Convergence Failure

**Error:** `Simulation failed to converge`

**Common causes:**
- Unrealistic component values
- Missing DC path to ground
- Positive feedback without limiting
- Discontinuities in behavioral models

**Solutions:**

```python
# Solution 1: Relax tolerance
circuit.add_directive(".options reltol=0.01")
circuit.add_directive(".options abstol=1e-10")

# Solution 2: Add initial conditions
circuit.add_directive(".ic V(out)=0")
circuit.add_directive(".nodeset V(bias)=2.5")

# Solution 3: Ensure DC path for capacitors
r_bias = Resistor("BIAS", "1G")  # 1 gigaohm
circuit.add(r_bias)
circuit.connect(r_bias.ports[0], Net("cap_node"))
circuit.connect(r_bias.ports[1], GND)

# Solution 4: Use GEAR integration
circuit.add_directive(".options method=gear")

# Solution 5: Increase iteration limits
circuit.add_directive(".options itl1=500")  # DC iterations
circuit.add_directive(".options itl4=100")  # Transient iterations

# Solution 6: Add GMIN
circuit.add_directive(".options gmin=1e-12")
```

### 5. Empty Results

**Symptom:** Simulation returns no data or all zeros.

**Causes and solutions:**

```python
# Cause 1: Missing probes
circuit.add_directive(".probe v(out)")  # Probe specific node
circuit.add_directive(".probe v(*)")    # Probe all nodes

# Cause 2: Analysis not capturing time range
# Wrong - signal settles at 2ms, simulation only 1ms
circuit.add_directive(".tran 1u 1m")
# Correct
circuit.add_directive(".tran 1u 10m")

# Cause 3: Wrong analysis type
# For frequency response, use AC not TRAN
circuit.add_directive(".ac dec 100 1 1G")
```

### 6. Engine Not Found

**Error:** `Engine 'ngspice' not found`

**Solutions:**

```bash
# macOS
brew install ngspice

# Ubuntu/Debian
sudo apt-get install ngspice

# Windows
choco install ngspice

# Or set environment variable
export SPICELAB_NGSPICE=/path/to/ngspice
```

### 7. Value Out of Range

**Warning:** Extreme component values detected

**Common issues:**
- Resistance too high (>1TΩ) or too low (<1mΩ)
- Capacitance too small (<1fF)
- Wrong unit prefix

**Check:**

```python
# Wrong - 1 millifarad is huge!
c = Capacitor("1", "1m")  # This is 1mF, not 1µF

# Correct
c = Capacitor("1", "1u")  # 1 microfarad
```

## Validation Before Simulation

Always validate circuits:

```python
from spicelab.validators import validate_circuit

result = validate_circuit(circuit)

if not result.is_valid:
    print("ERRORS:")
    for error in result.errors:
        print(f"  {error.message}")
        if error.suggestion:
            print(f"    Fix: {error.suggestion}")

if result.warnings:
    print("WARNINGS:")
    for warning in result.warnings:
        print(f"  [{warning.severity}] {warning.message}")
```

### Strict Mode

```python
# Treat warnings as errors
result = validate_circuit(circuit, strict=True)
```

## Debugging Workflow

### Step 1: Validate

```python
result = circuit.validate()
if not result.is_valid:
    # Fix errors first
    pass
```

### Step 2: Preview Netlist

```python
# Check generated netlist
print(circuit.build_netlist())

# Verify component values and connections
```

### Step 3: Run Simple Analysis First

```python
# Start with DC operating point
from spicelab.core.types import AnalysisSpec

op = AnalysisSpec("op", {})
handle = run_simulation(circuit, [op], engine="ngspice")
ds = handle.dataset()

# Check node voltages are reasonable
print(ds)
```

### Step 4: Add Complexity Gradually

```python
# Then try transient
tran = AnalysisSpec("tran", {"tstep": "1u", "tstop": "1m"})

# Use verbose mode
handle = run_simulation(circuit, [tran], engine="ngspice", verbose=True)
```

### Step 5: Use Dry Run

```python
# Validate without actually simulating
handle = run_simulation(circuit, [tran], engine="ngspice", dry_run=True)
# No simulation runs, but netlist is validated
```

## Error Code Reference

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
| E3001 | Parse | Netlist parse error |
| E4001 | Validation | Component validation |
| E4004 | Validation | Value out of range |

## Best Practices

1. **Validate early**: Run `validate()` before simulation
2. **Start simple**: Test with DC operating point first
3. **Use meaningful net names**: Easier to debug node voltages
4. **Check units**: Verify SI prefixes (u vs m vs n)
5. **Add diagnostics**: Use `.probe v(*)` during debugging
6. **Simplify first**: Test with ideal components before real models
7. **Check ground**: Every circuit needs a ground reference
8. **Save working versions**: Commit before major changes

## Getting Detailed Help

```python
from spicelab.exceptions import ConvergenceError

try:
    result = run_simulation(job)
except ConvergenceError as e:
    print(e)              # Basic message
    print(e.full_help())  # Detailed causes and solutions
    print(e.hints)        # Specific suggestions
```

## Exercises

### Exercise 10.1: Fix Floating Node
Given a circuit with floating node error, identify and fix the missing connection.

### Exercise 10.2: Convergence Fix
A BJT amplifier fails to converge. Apply appropriate fixes:
1. Add initial conditions
2. Relax tolerances
3. Add DC paths

### Exercise 10.3: Debug Empty Results
A simulation runs but returns empty data. Diagnose and fix:
1. Check analysis directives
2. Add probe commands
3. Verify time range

### Exercise 10.4: Unit Prefix Errors
Find and fix unit prefix mistakes in a filter circuit where the cutoff frequency is wrong by 1000x.

### Exercise 10.5: Validation Workflow
Create a pre-simulation validation function that:
1. Validates circuit
2. Checks for common issues
3. Suggests fixes
4. Only proceeds if valid

## Troubleshooting Checklist

- [ ] Circuit validates without errors
- [ ] All nodes have at least 2 connections (except outputs)
- [ ] Ground reference exists
- [ ] No voltage sources in parallel
- [ ] Component values are realistic
- [ ] Analysis parameters are appropriate
- [ ] Probe commands are present
- [ ] Time/frequency range covers behavior of interest

## Summary

| Problem | Quick Fix |
|---------|-----------|
| Floating node | Add missing connection or 1G resistor to GND |
| Missing ground | Connect source negative to GND |
| V-source loop | Add series resistance or remove duplicate |
| Convergence | Relax reltol, add .ic, use GEAR |
| Empty results | Add .probe, check time range |
| Engine not found | Install engine, set path |
| Wrong values | Check unit prefixes |

---

**Congratulations!** You've completed the SpiceLab tutorial series.

**See also:**
- [Troubleshooting Guide](../troubleshooting_guide.md) - Complete reference
- [Common Pitfalls](../common-pitfalls.md) - Avoid common mistakes
- [Examples](../../examples/) - More working examples
