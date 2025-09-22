# Examples Usage

This page highlights a few runnable examples and what they demonstrate.

Getting started (tutorial)
--------------------------

- `examples/getting_started.py`: a step-by-step tutorial that builds an RC circuit,
  runs AC, DC and TRAN analyses, and saves plots. Run with:

```bash
cd examples
uv run --active python getting_started.py
```

Monte Carlo analysis
--------------------

- `examples/monte_carlo_demo.py`: demonstrates running a Monte Carlo using a
  configurable runner. By default it uses a fast fake-runner for demos and CI.
  To run the example and save plots and an HTML report:

```bash
cd examples
uv run --active python monte_carlo_demo.py --n 50 --outdir ./mc_out
```

Register and metadata
---------------------

- `examples/register_and_metadata.py`: shows how to register a custom component
  with metadata that includes `.include` or `.model` directives, how to apply
  the metadata to a `Circuit`, and run a small operating-point analysis.

```bash
cd examples
uv run --active python register_and_metadata.py --outdir ./artifacts
```

More examples
-------------

See the `examples/` directory for additional demos: AC Bode plots, DC sweeps,
LTSpice roundtrip helpers, op-amp stability checks, and more.
