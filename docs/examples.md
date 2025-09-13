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
