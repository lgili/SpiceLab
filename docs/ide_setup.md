# SpiceLab IDE Setup Guide

This guide helps you configure your IDE for the best development experience with SpiceLab.

## Visual Studio Code (Recommended)

### 1. Install Required Extensions

Install these extensions for optimal SpiceLab development:

```
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
```

Or search in VS Code Extensions (Ctrl+Shift+X):
- **Python** (Microsoft)
- **Pylance** (Microsoft) - For type checking and autocomplete

Optional but helpful:
- **Python Docstring Generator** - For documentation
- **Error Lens** - Inline error display
- **GitLens** - Git integration

### 2. Configure Python Path

1. Open Command Palette (Ctrl+Shift+P)
2. Search "Python: Select Interpreter"
3. Choose the Python environment with SpiceLab installed

### 3. Enable Type Checking

Create or edit `.vscode/settings.json`:

```json
{
  "python.analysis.typeCheckingMode": "basic",
  "python.analysis.autoImportCompletions": true,
  "python.analysis.inlayHints.functionReturnTypes": true,
  "python.analysis.inlayHints.variableTypes": true,
  "[python]": {
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "ms-python.python"
  }
}
```

### 4. Use SpiceLab Snippets

The project includes pre-built snippets in `.vscode/spicelab.code-snippets`.

**Available snippets:**

| Prefix | Description |
|--------|-------------|
| `circuit` | New SpiceLab circuit |
| `spicelab-import` | Full import statement |
| `resistor` | Add resistor |
| `capacitor` | Add capacitor |
| `inductor` | Add inductor |
| `vdc` | DC voltage source |
| `vpulse` | Pulse voltage source |
| `diode` | Add diode |
| `opamp` | Ideal opamp |
| `rc-lowpass` | RC lowpass template |
| `butterworth` | Butterworth filter |
| `voltage-divider` | Voltage divider template |
| `inverting-amp` | Inverting amplifier |
| `validate` | Circuit validation |
| `monte-carlo` | Monte Carlo setup |
| `tran` | Transient analysis |
| `ac` | AC analysis |
| `e-series` | E-series resistor |
| `library` | Component library |
| `troubleshoot` | Troubleshooting |

**Usage:**
1. Type the prefix (e.g., `circuit`)
2. Press Tab to expand
3. Use Tab to navigate between placeholders

**Example:**
```python
# Type: circuit<Tab>
# Result:
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor
from spicelab.core.net import GND, Net

circuit = Circuit("my_circuit")

# Add components

# Build netlist
print(circuit.build_netlist())
```

### 5. Configure Tasks

Create `.vscode/tasks.json` for common operations:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Validate Circuit",
      "type": "shell",
      "command": "spicelab-validate ${file}",
      "group": "test",
      "problemMatcher": []
    },
    {
      "label": "Run Tests",
      "type": "shell",
      "command": "pytest tests/ -v",
      "group": "test",
      "problemMatcher": []
    },
    {
      "label": "Run Benchmarks",
      "type": "shell",
      "command": "pytest tests/benchmarks/ --benchmark-only",
      "group": "test",
      "problemMatcher": []
    }
  ]
}
```

Run tasks with Ctrl+Shift+B or Terminal > Run Task.

---

## PyCharm

### 1. Configure Interpreter

1. File > Settings > Project > Python Interpreter
2. Select or add the interpreter with SpiceLab installed
3. Verify SpiceLab appears in package list

### 2. Enable Type Checking

1. File > Settings > Editor > Inspections
2. Enable "Python > Type checker compatible"
3. Set severity level as desired

### 3. Configure Code Style

1. File > Settings > Editor > Code Style > Python
2. Set consistent formatting:
   - Indent: 4 spaces
   - Line length: 100 characters
   - Import sorting: isort compatible

### 4. Live Templates (Snippets)

Create custom live templates:

1. File > Settings > Editor > Live Templates
2. Create new group "SpiceLab"
3. Add templates:

**Example: Circuit Template**
- Abbreviation: `circuit`
- Template text:
```python
from spicelab.core.circuit import Circuit
from spicelab.core.components import $COMPONENT$
from spicelab.core.net import GND, Net

circuit = Circuit("$NAME$")

$END$
```
- Variables: Set `$COMPONENT$` and `$NAME$` as editable

### 5. External Tools

1. File > Settings > Tools > External Tools
2. Add SpiceLab validation:
   - Name: Validate Circuit
   - Program: `spicelab-validate`
   - Arguments: `$FilePath$`
   - Working directory: `$ProjectFileDir$`

---

## Jupyter / JupyterLab

SpiceLab works well in Jupyter notebooks for interactive circuit design.

### Setup

```bash
# Install Jupyter if needed
pip install jupyter jupyterlab

# Start JupyterLab
jupyter lab
```

### Recommended Extensions

For JupyterLab:
- **jupyterlab-lsp** - Language server for autocompletion
- **jupyterlab-git** - Git integration

```bash
pip install jupyterlab-lsp python-lsp-server
pip install jupyterlab-git
```

### Interactive Usage

```python
# In a Jupyter notebook cell
from spicelab import *
from spicelab.viz import plot_traces

# Create circuit
circuit = Circuit("rc_filter")
r1 = Resistor("R1", "10k")
c1 = Capacitor("C1", "100n")
v1 = Vpulse("V1", v1=0, v2=5, td="0", tr="1n", tf="1n", pw="1m", per="2m")

circuit.add(r1, c1, v1)
circuit.connect(v1.ports[0], Net("in"))
circuit.connect(v1.ports[1], GND)
circuit.connect(r1.ports[0], Net("in"))
circuit.connect(r1.ports[1], Net("out"))
circuit.connect(c1.ports[0], Net("out"))
circuit.connect(c1.ports[1], GND)

# Display netlist
print(circuit.build_netlist())

# Visualize (if simulation results available)
# plot_traces(result, signals=["v(out)"])
```

---

## Neovim / Vim

### Using with LSP

Install `pyright` or `pylsp` for type checking:

```bash
pip install pyright
# or
pip install python-lsp-server
```

Configure in your LSP setup (example for `nvim-lspconfig`):

```lua
require('lspconfig').pyright.setup{}
```

### Snippets with LuaSnip

Add SpiceLab snippets to your snippet configuration:

```lua
local ls = require("luasnip")
local s = ls.snippet
local t = ls.text_node
local i = ls.insert_node

ls.add_snippets("python", {
  s("spicelab-circuit", {
    t({"from spicelab.core.circuit import Circuit",
       "from spicelab.core.components import Resistor, Capacitor",
       "from spicelab.core.net import GND, Net",
       "",
       "circuit = Circuit(\""}), i(1, "my_circuit"), t({"\")"}),
  }),
})
```

---

## CLI Tools

SpiceLab provides command-line tools for common operations:

### spicelab-validate

Validate circuit files before simulation:

```bash
# Basic validation
spicelab-validate circuit.py

# Strict mode (warnings as errors)
spicelab-validate circuit.py --strict

# JSON output (for CI/CD)
spicelab-validate circuit.py --json

# Quiet mode (exit code only)
spicelab-validate circuit.py --quiet
```

### spicelab-troubleshoot

Diagnose circuit issues:

```bash
# Auto-diagnose
spicelab-troubleshoot circuit.py

# Focus on convergence issues
spicelab-troubleshoot circuit.py --convergence

# Interactive mode
spicelab-troubleshoot circuit.py --interactive

# Quick fix suggestions
spicelab-troubleshoot circuit.py --quick-fix
```

### spicelab doctor

Check installation and environment:

```bash
spicelab doctor
```

This checks:
- Python version
- Required packages
- SPICE engines (ngspice, ltspice, xyce)
- Environment variables

---

## Autocompletion Tips

### Best Practices for Autocompletion

1. **Use explicit imports:**
```python
# Good - autocomplete works well
from spicelab.core.components import Resistor
r = Resistor(  # Autocomplete shows parameters

# Less good - autocomplete may be limited
from spicelab import *
```

2. **Type hints help:**
```python
from spicelab.core.circuit import Circuit

def create_filter(fc: float) -> Circuit:
    circuit = Circuit("filter")
    # ...
    return circuit
```

3. **Use result types:**
```python
from spicelab.templates import butterworth_lowpass, FilterResult

result: FilterResult = butterworth_lowpass(fc=1000, order=4)
# Autocomplete shows: circuit, components, cutoff_frequency, q_factor
```

### Common Import Patterns

```python
# Core components
from spicelab import Circuit, Net, GND, Resistor, Capacitor, Inductor

# Voltage/current sources
from spicelab import Vdc, Vpulse, Vpwl, Idc, Ipulse

# Analysis
from spicelab import monte_carlo, NormalPct, UniformPct

# Templates
from spicelab.templates import rc_lowpass, butterworth_lowpass, voltage_divider

# Visualization
from spicelab.viz import plot_traces, plot_bode, monte_carlo_histogram

# Library components
from spicelab.library import create_component, list_components

# Validation
from spicelab.validators import validate_circuit

# Troubleshooting
from spicelab.troubleshooting import Troubleshooter, diagnose_circuit
```

---

## Troubleshooting IDE Issues

### "Module not found" errors

1. Verify SpiceLab is installed:
```bash
pip show spicelab
```

2. Check Python interpreter in IDE matches installation

3. Restart language server / IDE

### Autocomplete not working

1. Ensure Pylance/pyright is installed and active
2. Check `python.analysis.autoImportCompletions` is true
3. Verify `.vscode/settings.json` configuration

### Slow autocomplete

1. Exclude large directories:
```json
{
  "python.analysis.exclude": [
    "**/node_modules",
    "**/.git",
    "**/build"
  ]
}
```

2. Reduce type checking scope:
```json
{
  "python.analysis.typeCheckingMode": "off"
}
```

---

**Last Updated:** 2025-11-27

**See Also:**
- [Troubleshooting Guide](troubleshooting_guide.md) - Debug circuit issues
- [Template Catalog](template_catalog.md) - Available templates
- [Component Catalog](component_catalog.md) - Component reference
