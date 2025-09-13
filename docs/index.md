# PyCircuitKit

PyCircuitKit is a modern Python toolkit for building circuits, generating SPICE netlists, and running analyses (Monte Carlo, AC, transient) with lightweight testing helpers.

### Features

- Clean, typed APIs
- Pluggable SPICE runners (ngspice)
- Example-driven tutorials and automated tests

### Quick start

Run locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Try the examples in the `examples/` folder after installing the package. Many examples print SPICE cards you can save as `.cir` and run with Ngspice or LTspice.

### Simulate analog + ADC (quick note)

This repo includes an `AnalogMux8` component that emits SPICE for an 8:1 analog multiplexer and a Python `ADCModel` (in `src/cat/core/adc.py`) that emulates sample-and-hold, aperture/jitter and quantization. Typical workflow:

1. Use components' `spice_card(net_of)` to generate a netlist (mux, sources, R/C front-end).
2. Run Ngspice/LTspice transient simulation on the netlist.
3. Post-process the simulated node waveform in Python using `ADCModel.sample_from_function` or `sample_sh` to obtain digital codes.

### Quick links

- Installation: `installation.md`
- Getting started: `getting-started.md`
- Examples: `examples.md`
- Examples gallery: `examples-gallery.md`
