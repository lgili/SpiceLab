# Examples

This project ships runnable examples in `examples/`:

- `rc_tran.py` — transient of an RC low-pass
- `ac_bode.py` — RC Bode magnitude/phase plot
- `dc_sweep.py` — DC sweep example
- `step_sweep_grid.py` — Python grid over parameters + DataFrame
- `monte_carlo_rc.py` — Monte Carlo with metrics and plots
- `opamp_closed_loop.py` — simple op-amp closed-loop wiring
- `rc_highpass.py` — RC highpass netlist builder
- `import_ltspice_and_run.py` — import LTspice netlist and run

Run them as modules:

```bash
python -m examples.rc_tran
python -m examples.ac_bode
python -m examples.step_sweep_grid
python -m examples.monte_carlo_rc
```

See the `examples/` folder for details and more scripts.
