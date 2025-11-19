# M11: UX Enhancements

**Status:** Proposed
**Priority:** 🟡 MEDIUM
**Estimated Duration:** 10-12 weeks
**Dependencies:** M7 (measurements for viz), M8 (models for browsing), M3 (components)

## Problem Statement

SpiceLab currently has limited user experience features beyond basic Python API. There's no interactive CLI, no IDE integration, no visual tools for circuit building or result exploration, and no web-based interface for non-programmers or remote access.

### Current Gaps
- ❌ No interactive CLI (Rich/Textual)
- ❌ No Jupyter magic commands (`%%spice`)
- ❌ No VSCode extension (syntax highlighting, preview, autocomplete)
- ❌ No web UI (FastAPI + React)
- ❌ No notebook widgets (interactive plots, parameter sliders)
- ❌ No progress bars with ETA for long simulations
- ❌ No live simulation monitoring

### Impact
- **User Experience:** Steep learning curve, limited interactivity
- **Adoption:** Users prefer tools with better UX
- **Accessibility:** Non-programmers excluded
- **Productivity:** No visual feedback during development

## Objectives

1. **Interactive CLI** with Textual/Rich for visual circuit building
2. **Jupyter magic commands** (`%%spice`, `%load_circuit`, etc.)
3. **VSCode extension** (syntax highlighting, schematic preview, autocomplete)
4. **Web UI** (FastAPI + React) for circuit design and simulation
5. **Notebook widgets** (ipywidgets) for interactive parameter exploration
6. **Progress bars** with Rich for long simulations
7. **Live simulation monitoring** with websockets
8. **Target: VSCode extension 1.0, Web UI beta**

## Technical Design

### 1. Interactive CLI with Textual

```python
# spicelab/cli/interactive.py
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, DataTable, Button, Input, Static
from textual.containers import Container, Horizontal

class InteractiveCLI(App):
    """Interactive CLI for SpiceLab."""

    CSS = """
    Screen {
        background: $surface;
    }

    #circuit-view {
        height: 20;
        border: solid green;
    }

    #results-view {
        border: solid blue;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("Circuit Components:", id="circuit-view"),
            DataTable(id="components-table"),
            Horizontal(
                Button("Add Component", id="add-btn"),
                Button("Run Simulation", id="run-btn"),
                Button("View Results", id="results-btn"),
            ),
            Static("Simulation Results:", id="results-view"),
            DataTable(id="results-table"),
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-btn":
            self.run_simulation()
        # ... more handlers

    def run_simulation(self):
        # Run simulation and update results table
        pass

# Usage
if __name__ == "__main__":
    app = InteractiveCLI()
    app.run()
```

### 2. Jupyter Magic Commands

```python
# spicelab/jupyter/magic.py
from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
from IPython.display import display, HTML
import matplotlib.pyplot as plt

@magics_class
class SpiceLabMagics(Magics):
    """Jupyter magic commands for SpiceLab."""

    @line_magic
    def load_circuit(self, line):
        """Load circuit from netlist file.

        Usage: %load_circuit path/to/circuit.cir
        """
        from spicelab.core.circuit import Circuit

        circuit = Circuit.from_netlist_file(line.strip())
        self.shell.user_ns['circuit'] = circuit

        print(f"Loaded circuit: {circuit.name}")
        print(f"Components: {len(circuit.components)}")

        return circuit

    @cell_magic
    def spice(self, line, cell):
        """Run SPICE netlist in cell.

        Usage:
        %%spice
        * RC circuit
        V1 in 0 AC 1
        R1 in out 1k
        C1 out 0 100n
        .ac dec 100 1 1meg
        .end
        """
        from spicelab.core.circuit import Circuit
        from spicelab.api import run_simulation

        # Parse netlist from cell
        circuit = Circuit.from_netlist(cell)

        # Extract analyses from netlist
        analyses = self._extract_analyses(cell)

        # Run simulation
        result = run_simulation(circuit, analyses)

        # Store in namespace
        self.shell.user_ns['last_result'] = result

        # Plot results
        self._plot_results(result)

        return result

    @line_magic
    def sim_progress(self, line):
        """Show simulation progress bar."""
        # Enable progress bars for next simulation
        import spicelab.config
        spicelab.config.SHOW_PROGRESS = True

    def _plot_results(self, result):
        """Auto-plot common results."""
        ds = result.dataset()

        # AC analysis: Bode plot
        if 'frequency' in ds.dims:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

            for var in ds.data_vars:
                if var != 'frequency':
                    freq = ds.frequency.values
                    mag = 20 * np.log10(np.abs(ds[var].values))
                    phase = np.angle(ds[var].values, deg=True)

                    ax1.semilogx(freq, mag, label=var)
                    ax2.semilogx(freq, phase, label=var)

            ax1.set_ylabel("Magnitude (dB)")
            ax1.legend()
            ax1.grid(True)

            ax2.set_ylabel("Phase (deg)")
            ax2.set_xlabel("Frequency (Hz)")
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plt.show()

        # Transient: time plot
        elif 'time' in ds.dims:
            plt.figure(figsize=(10, 4))

            for var in ds.data_vars:
                if var != 'time':
                    plt.plot(ds.time.values, ds[var].values, label=var)

            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (V)")
            plt.legend()
            plt.grid(True)
            plt.show()

# Load extension
def load_ipython_extension(ipython):
    ipython.register_magics(SpiceLabMagics)
```

### 3. VSCode Extension

```json
// package.json
{
  "name": "spicelab-vscode",
  "displayName": "SpiceLab",
  "description": "SpiceLab integration for VSCode",
  "version": "1.0.0",
  "engines": {
    "vscode": "^1.70.0"
  },
  "categories": ["Programming Languages", "Visualization"],
  "activationEvents": [
    "onLanguage:spice",
    "onCommand:spicelab.runSimulation"
  ],
  "main": "./out/extension.js",
  "contributes": {
    "languages": [{
      "id": "spice",
      "aliases": ["SPICE", "spice"],
      "extensions": [".cir", ".sp", ".spi"],
      "configuration": "./language-configuration.json"
    }],
    "grammars": [{
      "language": "spice",
      "scopeName": "source.spice",
      "path": "./syntaxes/spice.tmLanguage.json"
    }],
    "commands": [{
      "command": "spicelab.runSimulation",
      "title": "SpiceLab: Run Simulation"
    }, {
      "command": "spicelab.previewSchematic",
      "title": "SpiceLab: Preview Schematic"
    }]
  }
}
```

```typescript
// src/extension.ts
import * as vscode from 'vscode';
import { spawn } from 'child_process';

export function activate(context: vscode.ExtensionContext) {
    // Register command: Run Simulation
    let runSimulation = vscode.commands.registerCommand('spicelab.runSimulation', () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }

        const document = editor.document;
        const circuitPath = document.fileName;

        // Run SpiceLab CLI
        const terminal = vscode.window.createTerminal('SpiceLab');
        terminal.sendText(`python -m spicelab.cli run ${circuitPath}`);
        terminal.show();
    });

    // Register command: Preview Schematic
    let previewSchematic = vscode.commands.registerCommand('spicelab.previewSchematic', () => {
        const panel = vscode.window.createWebviewPanel(
            'spicelab.schematic',
            'Circuit Schematic',
            vscode.ViewColumn.Two,
            {}
        );

        // Generate schematic SVG (using Python)
        const python = spawn('python', ['-m', 'spicelab.cli', 'schematic', '--svg']);

        let svgData = '';
        python.stdout.on('data', (data) => {
            svgData += data.toString();
        });

        python.on('close', () => {
            panel.webview.html = getWebviewContent(svgData);
        });
    });

    context.subscriptions.push(runSimulation, previewSchematic);
}

function getWebviewContent(svgData: string): string {
    return `
<!DOCTYPE html>
<html>
<head>
    <style>
        body { margin: 0; padding: 20px; }
        svg { max-width: 100%; }
    </style>
</head>
<body>
    ${svgData}
</body>
</html>
    `;
}
```

### 4. Web UI with FastAPI + React

```python
# spicelab/web/api.py
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio

app = FastAPI(title="SpiceLab Web API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class CircuitDefinition(BaseModel):
    name: str
    netlist: str
    analyses: list[dict]

@app.post("/api/simulate")
async def run_simulation(circuit_def: CircuitDefinition):
    """Run simulation and return results."""
    from spicelab.core.circuit import Circuit
    from spicelab.api import run_simulation

    # Parse netlist
    circuit = Circuit.from_netlist(circuit_def.netlist)

    # Run simulation (async)
    result = run_simulation(circuit, circuit_def.analyses)

    # Convert to JSON-serializable format
    dataset = result.dataset()
    return {
        "success": True,
        "data": dataset.to_dict(),
    }

@app.websocket("/ws/simulate")
async def simulate_websocket(websocket: WebSocket):
    """Live simulation updates via WebSocket."""
    await websocket.accept()

    while True:
        # Receive circuit definition
        data = await websocket.receive_json()

        # Run simulation with progress updates
        for progress in run_simulation_with_progress(data):
            await websocket.send_json({
                "type": "progress",
                "value": progress,
            })

        await websocket.send_json({
            "type": "complete",
            "result": "..."
        })
```

```tsx
// web-ui/src/components/CircuitEditor.tsx
import React, { useState } from 'react';
import { Editor } from '@monaco-editor/react';
import axios from 'axios';

export function CircuitEditor() {
    const [netlist, setNetlist] = useState('* RC Circuit\nV1 in 0 AC 1\n...');
    const [results, setResults] = useState(null);

    const runSimulation = async () => {
        const response = await axios.post('http://localhost:8000/api/simulate', {
            name: 'circuit',
            netlist: netlist,
            analyses: [{ type: 'ac', params: { ... } }],
        });

        setResults(response.data.data);
    };

    return (
        <div className="circuit-editor">
            <Editor
                height="400px"
                language="spice"
                value={netlist}
                onChange={(value) => setNetlist(value || '')}
            />
            <button onClick={runSimulation}>Run Simulation</button>
            {results && <ResultsViewer data={results} />}
        </div>
    );
}
```

### 5. Jupyter Widgets (ipywidgets)

```python
# spicelab/jupyter/widgets.py
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt

class CircuitParameterExplorer:
    """Interactive parameter exploration widget."""

    def __init__(self, circuit, analyses, parameter_ranges):
        self.circuit = circuit
        self.analyses = analyses
        self.parameter_ranges = parameter_ranges

        # Create sliders
        self.sliders = {}
        for param_name, (min_val, max_val) in parameter_ranges.items():
            slider = widgets.FloatSlider(
                value=(min_val + max_val) / 2,
                min=min_val,
                max=max_val,
                step=(max_val - min_val) / 100,
                description=param_name,
                continuous_update=False,
            )
            slider.observe(self._on_parameter_change, names='value')
            self.sliders[param_name] = slider

        # Output widget
        self.output = widgets.Output()

        # Initial plot
        self._update_plot()

    def display(self):
        """Display widget."""
        slider_box = widgets.VBox(list(self.sliders.values()))
        display(widgets.HBox([slider_box, self.output]))

    def _on_parameter_change(self, change):
        """Handle parameter change."""
        self._update_plot()

    def _update_plot(self):
        """Update plot with current parameters."""
        # Update circuit
        for param_name, slider in self.sliders.items():
            comp_ref, attr = param_name.split('.')
            comp = self.circuit.get_component(comp_ref)
            setattr(comp, attr, slider.value)

        # Run simulation
        result = run_simulation(self.circuit, self.analyses)
        ds = result.dataset()

        # Plot
        with self.output:
            self.output.clear_output(wait=True)
            plt.figure(figsize=(8, 4))

            if 'frequency' in ds.dims:
                # Bode plot
                for var in ds.data_vars:
                    if var != 'frequency':
                        mag_db = 20 * np.log10(np.abs(ds[var].values))
                        plt.semilogx(ds.frequency.values, mag_db, label=var)
            else:
                # Time plot
                for var in ds.data_vars:
                    if var != 'time':
                        plt.plot(ds.time.values, ds[var].values, label=var)

            plt.legend()
            plt.grid(True)
            plt.show()

# Usage in Jupyter
explorer = CircuitParameterExplorer(
    circuit=circuit,
    analyses=[ac_analysis],
    parameter_ranges={
        'R1.resistance': (100, 10_000),
        'C1.capacitance': (1e-9, 1e-6),
    }
)
explorer.display()
```

### 6. Progress Bars with Rich

```python
# spicelab/progress.py
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

class SimulationProgress:
    """Progress bar for simulations."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def __enter__(self):
        if self.enabled:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            )
            self.progress.__enter__()
        return self

    def __exit__(self, *args):
        if self.enabled:
            self.progress.__exit__(*args)

    def add_task(self, description: str, total: int):
        """Add progress task."""
        if self.enabled:
            return self.progress.add_task(description, total=total)
        return None

    def update(self, task_id, advance: int = 1):
        """Update progress."""
        if self.enabled and task_id is not None:
            self.progress.update(task_id, advance=advance)

# Usage
with SimulationProgress() as progress:
    task = progress.add_task("Running Monte Carlo", total=1000)

    for i in range(1000):
        # Run simulation
        progress.update(task)
```

## Implementation Plan

### Week 1-2: Interactive CLI (Textual)
### Week 3-4: Jupyter Magic Commands
### Week 5-6: VSCode Extension (MVP)
### Week 7-8: Web UI Backend (FastAPI)
### Week 9-10: Web UI Frontend (React)
### Week 11: Jupyter Widgets
### Week 12: Progress Bars & Documentation

## Success Metrics

- [ ] Interactive CLI functional
- [ ] Jupyter magics working (`%%spice`, `%load_circuit`)
- [ ] VSCode extension published (1.0)
- [ ] Web UI beta deployed
- [ ] Jupyter widgets interactive
- [ ] Progress bars in all long operations
- [ ] User satisfaction >4.5/5

## Dependencies

- M7 (measurements for visualization)
- M8 (models for browsing)
- Textual, Rich
- ipywidgets
- FastAPI, React
- VSCode API

## References

- [Textual](https://textual.textualize.io/)
- [IPython Magics](https://ipython.readthedocs.io/en/stable/config/custommagics.html)
- [VSCode Extension API](https://code.visualstudio.com/api)
- [FastAPI](https://fastapi.tiangolo.com/)
