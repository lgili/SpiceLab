# Examples

This page describes the runnable examples included in the `examples/` folder
and shows common CLI flags used by many scripts.

Overview
--------

The repository includes a set of examples that demonstrate typical
workflows. Each example is a self-contained Python script that accepts
optional CLI flags documented below.

- `getting_started.py` — step-by-step tutorial: build a circuit and run AC,
  DC and transient analyses.
- `monte_carlo_demo.py` — Monte Carlo sweep, computes metrics and writes an
  HTML report. Supports a fake-runner for CI and a real-run option with
  `--real-run`.
- `register_and_metadata.py` — shows how to register custom components and
  attach metadata to them; writes a small models file and runs an OP analysis
  to demonstrate the model usage.

Common CLI flags
----------------

Most examples accept these optional flags (use `-h` to see them in each
script):

- `--outdir PATH` — write plots, model files and reports to PATH (defaults to
  the current working directory). Use a temporary directory in CI to avoid
  polluting the repo.
- `--real-run` — when present, the script will attempt to call the real
  simulator (`ngspice`) instead of using the fast fake-runner. Omit this in
  CI if ngspice is not available.

Fake-runner vs real-run
-----------------------

The examples are written to accept either a "fake" runner (fast, deterministic
for CI) or to invoke the real external simulator. Prefer the fake-runner in
unit tests and CI. When debugging actual simulator behavior, run with
`--real-run`.
# Examples

This project ships runnable examples in `examples/`:

- `rc_tran.py` — transient of an RC low-pass
- `ac_bode.py` — RC Bode magnitude/phase plot
- `dc_sweep.py` — DC sweep example
- `step_sweep_grid.py` — Python grid over parameters + DataFrame
- `monte_carlo_rc.py` — Monte Carlo with metrics and plots
- `opamp_closed_loop.py` — simple op-amp closed-loop wiring
- `opamp_stability.py` — op-amp closed-loop Bode and margins
- `pt1000_mc.py` — PT1000 front-end Monte Carlo (temp error histogram)
- `rc_highpass.py` — RC highpass netlist builder
- `import_ltspice_and_run.py` — import LTspice netlist and run
- `ltspice_schematic_roundtrip.py` — export/import `.asc` schematics using `cat.io.ltspice_asc`
- `analog_mux_demo.py` — example showing AnalogMux8 netlist and DOT diagram

Run them as modules:

```bash
python -m examples.rc_tran
python -m examples.ac_bode
python -m examples.step_sweep_grid
python -m examples.monte_carlo_rc
```

See the `examples/` folder for details and more scripts.

Analog multiplexer example
--------------------------

The repository includes an `AnalogMux8` component (1-to-8 analog multiplexer).
See `examples/analog_mux_demo.py` for a runnable demonstration that prints the
generated netlist and emits a Graphviz DOT string for quick visual inspection.

Notes:
- The component supports static selection via the `sel` parameter or dynamic
	control by enabling `enable_ports=True` which exposes `en0..en7` control
	pins and emits `S...` switches in the netlist. If `emit_model=True` the
	example will include a recommended `.model` line for the switches.
- Default series resistance `r_series` is 100 (ohms). Off-resistance is `1G`.

Run the demo
------------

To run the example locally (from the project root):

```bash
PYTHONPATH=src python examples/analog_mux_demo.py
```

This prints the generated SPICE card and a DOT string which you can render
with Graphviz if available.

Quick snippet
-------------

```python
from cat.core.components import AnalogMux8

mux = AnalogMux8(ref="MU1", r_series=100, sel=4)
print(mux.spice_card(lambda p: p.name))
```
