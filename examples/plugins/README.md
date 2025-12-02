# Plugin Examples

This directory contains practical examples demonstrating how to use the SpiceLab plugin system.

## Available Examples

### 01_report_generator.py
**Auto-generate simulation reports in Markdown, HTML, or JSON formats.**

Features demonstrated:
- Manual report generation
- Auto-report via hooks
- Configuration options
- Different output formats

```python
from spicelab.plugins.examples import ReportGeneratorPlugin

plugin = ReportGeneratorPlugin()
plugin.configure({"output_dir": "./reports", "format": "html"})
path = plugin.generate_report(circuit, result)
```

### 02_design_rules.py
**Design Rule Checks (DRC) for circuit validation.**

Features demonstrated:
- Basic DRC checks
- Configuring rules
- Component value validation
- Severity levels
- Blocking simulation on errors

```python
from spicelab.plugins.examples import DesignRulesPlugin

plugin = DesignRulesPlugin()
result = plugin.check(circuit)
if not result.passed:
    for violation in result.violations:
        print(violation)
```

### 03_circuit_templates.py
**Pre-built circuit templates for common designs.**

Available templates:
- **Filters**: RC/RL/RLC lowpass, highpass, bandpass
- **Basic**: Voltage dividers
- **Amplifiers**: Inverting, non-inverting, voltage follower
- **Oscillators**: RC phase-shift
- **Test**: Transient test circuits

```python
from spicelab.plugins.examples import CircuitTemplatesPlugin

plugin = CircuitTemplatesPlugin()
circuit = plugin.create("rc_lowpass", cutoff_freq=1000)
circuit = plugin.create("voltage_divider", r1="10k", r2="10k", vin=5.0)
```

### 04_simulation_profiler.py
**Profile simulation performance and identify bottlenecks.**

Features demonstrated:
- Basic profiling
- Statistics analysis
- Cache tracking
- Finding slowest operations
- Exporting profile data
- Memory usage tracking

```python
from spicelab.plugins.examples import SimulationProfilerPlugin

plugin = SimulationProfilerPlugin()
plugin.activate()
# ... run simulations ...
print(plugin.get_report())
plugin.export_profile("profile.json")
```

### 05_auto_backup.py
**Automatic circuit backup with version history.**

Features demonstrated:
- Manual and automatic backups
- Version history
- Deduplication
- Compression
- Restore from backup
- Backup management

```python
from spicelab.plugins.examples import AutoBackupPlugin

plugin = AutoBackupPlugin()
plugin.configure({"backup_dir": "./backups", "max_backups": 50})
plugin.activate()
# Backups happen automatically before simulations
```

## Running Examples

```bash
# Run individual examples
python examples/plugins/01_report_generator.py
python examples/plugins/02_design_rules.py
python examples/plugins/03_circuit_templates.py
python examples/plugins/04_simulation_profiler.py
python examples/plugins/05_auto_backup.py
```

## Quick Start

All plugins follow a similar pattern:

```python
from spicelab.plugins.examples import <PluginName>

# 1. Create instance
plugin = <PluginName>()

# 2. Configure (optional)
plugin.configure({...})

# 3. Activate (registers hooks)
plugin.activate()

# 4. Use the plugin
# ...

# 5. Deactivate when done
plugin.deactivate()
```

## Using with PluginManager

For production use, register plugins with the PluginManager:

```python
from spicelab.plugins import PluginManager
from spicelab.plugins.examples import ReportGeneratorPlugin, DesignRulesPlugin

manager = PluginManager()

# Load plugins
manager.loader.load_from_class(ReportGeneratorPlugin)
manager.loader.load_from_class(DesignRulesPlugin)

# Activate
manager.activate_plugin("report-generator")
manager.activate_plugin("design-rules")

# Configure
manager.set_plugin_settings("report-generator", {
    "output_dir": "./reports",
    "format": "html",
})
```

## Creating Custom Plugins

See the plugin source code in `spicelab/plugins/examples/` for implementation details. Each plugin demonstrates:

- Proper metadata definition
- Hook registration
- Configuration handling
- Activation/deactivation lifecycle
