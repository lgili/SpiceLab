# Analog multiplexer demo

The `AnalogMux8` component still supports enable ports and `.model` emission.
For the latest example run `examples/analog_mux_demo.py`:

```bash
uv run --active python examples/analog_mux_demo.py
```

The script prints the SPICE card and, if Graphviz is available, exports an SVG
showing the mux topology.
