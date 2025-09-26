# Troubleshooting

- **Engine not found** – install the simulator and ensure it is on PATH, or set
  `SPICELAB_NGSPICE`, `SPICELAB_LTSPICE`, `SPICELAB_XYCE` to the absolute binary path.
- **Binary RAW detected** – `load_dataset` defaults to ASCII. Re-run the engine via
  the orchestrator (which requests ASCII) or opt in with `allow_binary=True`.
- **Unconnected port** – every component port must be tied to a net before calling
  `Circuit.build_netlist()`.
- **Trace missing from dataset** – canonical names are `V(node)` and `I(element)`;
  check `handle.dataset().data_vars` for the normalised names.
- **Phase / magnitude confusion** – feed AC datasets into the measurement helpers
  or Plotly utilities (`plot_bode`). Use `complex_components=True` when loading
  RAW files if you need real/imaginary channels.
- **Cache not reused** – ensure `cache_dir` is stable and that the circuit hash
  incorporates any external `.model` or `.include` files.
