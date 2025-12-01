# Chapter 9: Integration & Automation

This chapter covers integrating SpiceLab into development workflows: CI/CD, batch processing, and Jupyter notebooks.

## Command-Line Interface

### Validate Circuits

```bash
# Validate a circuit file
spicelab-validate circuit.py

# Strict mode (warnings as errors)
spicelab-validate circuit.py --strict

# JSON output for parsing
spicelab-validate circuit.py --json

# Quiet mode (exit code only)
spicelab-validate circuit.py --quiet
```

### Troubleshoot Circuits

```bash
# Auto-diagnose issues
spicelab-troubleshoot circuit.py

# Focus on convergence
spicelab-troubleshoot circuit.py --convergence

# Interactive mode
spicelab-troubleshoot circuit.py --interactive
```

### Check Environment

```bash
# Verify installation
spicelab doctor
```

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/circuit-tests.yml
name: Circuit Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install spicelab pytest
        sudo apt-get install -y ngspice

    - name: Validate circuits
      run: |
        for file in circuits/*.py; do
          spicelab-validate "$file" --strict
        done

    - name: Run tests
      run: pytest tests/ -v
```

### Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: validate-circuits
        name: Validate SPICE circuits
        entry: spicelab-validate
        language: python
        files: \.py$
        args: [--strict]
```

### GitLab CI Example

```yaml
# .gitlab-ci.yml
circuit-validation:
  image: python:3.11
  before_script:
    - pip install spicelab
    - apt-get update && apt-get install -y ngspice
  script:
    - spicelab-validate circuits/*.py --strict
  rules:
    - changes:
        - circuits/*.py
```

## Batch Processing

### Process Multiple Files

```python
"""Batch process circuit files"""
import glob
from pathlib import Path

from spicelab.core.types import AnalysisSpec
from spicelab.engines import run_simulation
from spicelab.validators import validate_circuit

def process_circuit_file(filepath: str):
    """Load and simulate a circuit from file."""
    # Execute the file to get circuit object
    namespace = {}
    exec(Path(filepath).read_text(), namespace)
    circuit = namespace.get("circuit")

    if circuit is None:
        raise ValueError(f"No 'circuit' object in {filepath}")

    # Validate
    result = validate_circuit(circuit)
    if not result.is_valid:
        return {"file": filepath, "status": "invalid", "errors": result.errors}

    # Simulate
    analyses = [AnalysisSpec("op", {})]
    handle = run_simulation(circuit, analyses, engine="ngspice")
    ds = handle.dataset()

    return {"file": filepath, "status": "success", "results": ds}

# Process all circuit files
results = []
for filepath in glob.glob("circuits/*.py"):
    try:
        result = process_circuit_file(filepath)
        results.append(result)
        print(f"✓ {filepath}")
    except Exception as e:
        results.append({"file": filepath, "status": "error", "error": str(e)})
        print(f"✗ {filepath}: {e}")

# Summary
passed = sum(1 for r in results if r["status"] == "success")
print(f"\n{passed}/{len(results)} circuits passed")
```

### Parallel Batch Processing

```python
from concurrent.futures import ProcessPoolExecutor
import glob

def process_file(filepath):
    """Process single file (for parallel execution)."""
    try:
        result = process_circuit_file(filepath)
        return filepath, "success", result
    except Exception as e:
        return filepath, "error", str(e)

# Process in parallel
files = glob.glob("circuits/*.py")
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_file, files))

for filepath, status, data in results:
    print(f"{filepath}: {status}")
```

## Jupyter Notebooks

### Basic Usage

```python
# In a Jupyter cell
from spicelab import Circuit, Net, GND, Resistor, Vdc
from spicelab.viz import plot_traces

# Build circuit
circuit = Circuit("notebook_demo")
# ... add components ...

# Run and plot
handle = run_simulation(circuit, analyses, engine="ngspice")
plot_traces(handle.dataset(), signals=["V(out)"])
```

### Interactive Widgets

```python
from ipywidgets import interact, FloatSlider
import matplotlib.pyplot as plt

def simulate_and_plot(resistance):
    """Interactive simulation with slider."""
    circuit = Circuit("interactive")
    r1 = Resistor("1", resistance)
    # ... build circuit ...

    handle = run_simulation(circuit, analyses, engine="ngspice")
    ds = handle.dataset()

    plt.figure(figsize=(10, 4))
    plt.plot(ds["time"].values * 1e3, ds["V(out)"].values)
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (V)")
    plt.title(f"R = {resistance/1000:.1f}k")
    plt.show()

# Create interactive slider
interact(
    simulate_and_plot,
    resistance=FloatSlider(min=1000, max=100000, step=1000, value=10000)
)
```

### Export Notebook Results

```python
# Save results to file
ds.to_netcdf("simulation_results.nc")

# Export plots
import matplotlib.pyplot as plt
plt.savefig("frequency_response.png", dpi=150)
plt.savefig("frequency_response.pdf")
```

## Report Generation

### Automated Reports

```python
"""Generate simulation report"""
from datetime import datetime
from pathlib import Path

def generate_report(circuit, results, output_path: str):
    """Generate markdown report from simulation results."""
    report = []
    report.append(f"# Simulation Report: {circuit.name}")
    report.append(f"\nGenerated: {datetime.now().isoformat()}")

    # Circuit info
    report.append("\n## Circuit")
    report.append(f"- Components: {len(circuit.components)}")
    report.append(f"- Nets: {len(circuit.nets)}")

    # Netlist
    report.append("\n## Netlist")
    report.append("```spice")
    report.append(circuit.build_netlist())
    report.append("```")

    # Results
    report.append("\n## Results")
    for key, value in results.items():
        report.append(f"- {key}: {value}")

    # Write file
    Path(output_path).write_text("\n".join(report))

# Use
generate_report(circuit, {"V(out)": 5.0, "I(V1)": 1e-3}, "report.md")
```

### HTML Reports with Plots

```python
import base64
from io import BytesIO

def figure_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_html_report(circuit, ds, output_path: str):
    """Generate HTML report with embedded plots."""
    import matplotlib.pyplot as plt

    # Create plot
    fig, ax = plt.subplots()
    ax.plot(ds["time"].values * 1e3, ds["V(out)"].values)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (V)")
    img_b64 = figure_to_base64(fig)
    plt.close(fig)

    html = f"""
    <html>
    <head><title>{circuit.name} Report</title></head>
    <body>
    <h1>{circuit.name}</h1>
    <h2>Time Domain Response</h2>
    <img src="data:image/png;base64,{img_b64}" />
    <h2>Netlist</h2>
    <pre>{circuit.build_netlist()}</pre>
    </body>
    </html>
    """

    Path(output_path).write_text(html)

generate_html_report(circuit, ds, "report.html")
```

## Testing Patterns

### Unit Tests for Circuits

```python
"""tests/test_voltage_divider.py"""
import pytest
from spicelab import Circuit, Resistor, Vdc, Net, GND
from spicelab.core.types import AnalysisSpec
from spicelab.engines import run_simulation

def create_voltage_divider(r1_value, r2_value, vin):
    """Create voltage divider circuit."""
    circuit = Circuit("test_divider")
    v1 = Vdc("1", vin)
    r1 = Resistor("1", r1_value)
    r2 = Resistor("2", r2_value)

    circuit.add(v1, r1, r2)
    circuit.connect(v1.ports[0], Net("vin"))
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r1.ports[0], Net("vin"))
    circuit.connect(r1.ports[1], Net("vout"))
    circuit.connect(r2.ports[0], Net("vout"))
    circuit.connect(r2.ports[1], GND)

    return circuit

class TestVoltageDivider:
    def test_50_percent_division(self):
        """Test 50% voltage divider."""
        circuit = create_voltage_divider("10k", "10k", 10.0)
        handle = run_simulation(circuit, [AnalysisSpec("op", {})], engine="ngspice")
        ds = handle.dataset()

        v_out = float(ds["V(vout)"].values)
        assert pytest.approx(v_out, rel=0.01) == 5.0

    def test_33_percent_division(self):
        """Test 1/3 voltage divider."""
        circuit = create_voltage_divider("20k", "10k", 9.0)
        handle = run_simulation(circuit, [AnalysisSpec("op", {})], engine="ngspice")
        ds = handle.dataset()

        v_out = float(ds["V(vout)"].values)
        assert pytest.approx(v_out, rel=0.01) == 3.0
```

### Regression Tests

```python
"""tests/test_regression.py"""
import json
import pytest
from pathlib import Path

# Load expected results
EXPECTED = json.loads(Path("tests/expected_results.json").read_text())

@pytest.mark.parametrize("circuit_name,expected", EXPECTED.items())
def test_circuit_output(circuit_name, expected):
    """Regression test against expected values."""
    # Load circuit
    circuit = load_circuit(f"circuits/{circuit_name}.py")

    # Simulate
    handle = run_simulation(circuit, [AnalysisSpec("op", {})], engine="ngspice")
    ds = handle.dataset()

    # Compare
    for node, value in expected.items():
        actual = float(ds[node].values)
        assert pytest.approx(actual, rel=0.05) == value, \
            f"{circuit_name}: {node} expected {value}, got {actual}"
```

## Exercises

### Exercise 9.1: CI Pipeline
Create a GitHub Actions workflow that:
1. Validates all circuits in `circuits/`
2. Runs pytest
3. Generates a coverage report

### Exercise 9.2: Batch Report
Write a script that processes all circuits in a directory and generates:
1. Summary CSV with key measurements
2. Individual HTML reports with plots

### Exercise 9.3: Interactive Notebook
Create a Jupyter notebook that:
1. Builds an RC filter with sliders for R and C
2. Shows real-time frequency response
3. Exports results to CSV

### Exercise 9.4: Regression Suite
Set up regression tests for 5 circuits with expected values.
Run with pytest and generate a report.

### Exercise 9.5: Pre-commit Hook
Configure pre-commit to validate circuits before each commit.
Test that invalid circuits block commits.

## Integration Checklist

| Integration | Tools | Purpose |
|-------------|-------|---------|
| CI/CD | GitHub Actions, GitLab CI | Automated testing |
| Pre-commit | pre-commit | Validate before commit |
| Notebooks | Jupyter, JupyterLab | Interactive exploration |
| Reporting | Markdown, HTML | Documentation |
| Testing | pytest | Regression testing |

## Next Steps

- [Chapter 10: Troubleshooting](10_troubleshooting.md) - Debugging techniques

---

**See also:**
- [CLI Reference](../cli-ci.md) - Command-line tools
- [IDE Setup](../ide_setup.md) - Development environment
