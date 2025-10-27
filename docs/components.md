# Components Overview

spicelab ships a typed component library. Each component exposes
`ports`, participates in the circuit connectivity graph, and knows how to format
its SPICE card when you call `Circuit.build_netlist()`.

```python
from spicelab.core.circuit import Circuit
from spicelab.core.components import Vdc, Resistor, Capacitor
from spicelab.core.net import GND

c = Circuit("rc")
V1 = Vdc("VIN", 5.0)
R1 = Resistor("R", "1k")
C1 = Capacitor("C", "100n")
for comp in (V1, R1, C1):
    c.add(comp)
c.connect(V1.ports[0], R1.ports[0])
c.connect(R1.ports[1], C1.ports[0])
c.connect(V1.ports[1], GND)
c.connect(C1.ports[1], GND)
print(c.build_netlist())
```

## CircuitBuilder DSL

For larger circuits the iterative `add`/`connect` dance can get noisy. The
`CircuitBuilder` DSL shipped in `spicelab.dsl.builder` lets you declare nets,
drop parts, and wire busses inline:

```python
from spicelab.dsl import CircuitBuilder

builder = CircuitBuilder("pi_filter")
VIN, VOUT, GND = builder.net("VIN"), builder.net("VOUT"), builder.gnd()
bus = builder.bus("VIN", "MID", "VOUT")

builder.vdc("VIN", VIN, GND, 5.0)
builder.resistor("R1", VIN, bus.MID, "220")
builder.capacitor("C1", bus.MID, GND, "10u")
builder.inductor("L1", bus.MID, VOUT, "47u")
builder.capacitor("C2", VOUT, GND, "22u")

circuit = builder.circuit
print(circuit.build_netlist())
```

Nets can be named, duplicated, or grouped in busses. Every shorthand accepts
either explicit reference designators (`"R1"`) or lets the builder auto-number
using prefixes (`builder.resistor("R", ...)`).

### Circuit context DSL (parameters & directives)

For scripts that lean heavily on `.param`, `.option`, and other control
statements, try the lightweight context-based DSL:

```python
from spicelab.dsl import Circuit, Net, Param, Option, TEMP, IC, Directive, R, V

with Circuit("rc_control") as ctx:
    vin = Net("vin")
    vout = Net("vout")
    gnd = Net("0")

    Param("Rval", "10k")                 # -> .param Rval=10k
    Option(reltol=1e-3, abstol=1e-6)       # -> .option reltol=0.001 abstol=1e-06
    TEMP(27, 85)                           # -> .temp 27 85
    IC(vout="0")                          # -> .ic V(vout)=0
    Directive(".save V(vout)")            # raw escape hatch (validated)

    V("VIN", vin, gnd, 5.0)
    R("R1", vin, vout, "Rval")

circuit = ctx.circuit
print(circuit.build_netlist())
```

All helpers validate the expressions before emitting them into the netlist. In
particular `Param`, `Option`, `TEMP`, and `IC` use a safe expression normaliser
that accepts numbers, engineering suffixes (`10k`, `100n`), basic math symbols,
and references to previously declared parameters. The `Directive` helper keeps a
“safe” mode on by default—lines must start with a dot. Pass `safe=False` to drop
the guard when you really need arbitrary text.

The context manager stores the underlying `spicelab.core.circuit.Circuit` in
`ctx.circuit`, so you can continue using all low-level APIs. Component
shortcuts (`R`, `C`, `L`, `V`) simply instantiate the typed components and wire
them to the given nets (strings or `Net` objects). Use `spicelab.dsl.place(...)`
to register any other component manually.

## Common parts

| Component | Helper | Notes |
|-----------|--------|-------|
| `Resistor(ref, value)` | `R(value)` | Value accepts numbers or suffixed strings (`"1k"`, `"10m"`). |
| `Capacitor(ref, value)` | `C(value)` | Two-terminal capacitor. |
| `Inductor(ref, value)` | `L(value)` | Two-terminal inductor. |
| `Vdc`, `Idc` | `V`, `I` | Independent DC sources. |
| `Vpulse`, `Ipulse` | `VP`, `IP` | Pulse sources. |
| `Vsin`, `Isin` | `VSIN`, `ISIN` | Sine sources. |
| `Vpwl`, `Ipwl` | `VPWL`, `IPWL` | Piecewise-linear sources. |
| `VA`, `IA` | Small-signal AC helpers. |
| `VCVS`, `VCCS`, `CCCS`, `CCVS` | `E`, `G`, `F`, `H` | Controlled sources. |
| `OpAmpIdeal` | `OA` | Three-port ideal op-amp (VCVS). |
| `AnalogMux8` | – | 8:1 analog multiplexer with optional enable pins. |

All helpers live in `spicelab.core.components`. Import only what you need, or
use the shorthand constructors (`R`, `C`, `V`, etc.) for quick sketches.

## Directives & models

When you need SPICE directives (e.g. `.model`, `.include`) call
`circuit.add_directive(...)`. Components such as `Diode` or `AnalogMux8`
expose convenience booleans to emit models automatically; check their docstrings
for arguments.

Example:
```python
from spicelab.core.components import Diode

D1 = Diode("D1", model="DFAST")
c.add_directive(".model DFAST D(Is=1e-14 Rs=0.5)")
```

## Visual preview

Use `Circuit.summary()` to inspect connectivity and `Circuit.render_svg()` to
export a quick Graphviz diagram (see `examples/circuit_preview.py`).

`Circuit.summary_table()` and `Circuit.connectivity_dataframe()` expose the same
data as a plain-text table or pandas `DataFrame`. These helpers power the
interactive widgets below and make it easy to drop connectivity checks into a
notebook or CLI report.

## Notebook helpers

Install the `viz` extra to bring in Plotly and ipywidgets:

```bash
pip install circuit-toolkit[viz]
```

From a notebook you can embed circuit tables and plot Monte Carlo datasets via
`spicelab.viz.notebook`:

```python
from spicelab.viz.notebook import connectivity_widget, dataset_plot_widget

connectivity_widget(circuit)  # interactive component/net browser

fig_widget = dataset_plot_widget(dataset)
fig_widget  # display a FigureWidget with selector dropdowns
```

Widgets fall back gracefully when optional dependencies are missing and play
nicely alongside the DSL builder for rapid prototyping inside VS Code or Jupyter.

For full workflows (sweeps, Monte Carlo, engines) read the dedicated guides –
components slot directly into those orchestration helpers.
