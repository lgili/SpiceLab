## Getting started

This quickstart guides a new user through installation, running a simple
example, and best practices for CI-friendly execution. It links to example
pages and the API reference.

1) Create an environment and install
------------------------------------

Use an isolated environment. Below is an example using the repository's
helper `uv` (short for uenv). Replace it with `python -m venv` or `conda`
if you prefer.

```bash
# from the repository root
uv run --active pip install -e .
uv run --active pip install -r dev-requirements.txt
```

The development requirements will pull in test and docs dependencies used for
the examples and local site builds.

2) Run the quick example
------------------------

The `examples/getting_started.py` script walks through building a small RC
circuit and running AC, DC and transient analyses. To run it and collect
artifacts into a dedicated folder:

```bash
cd examples
python getting_started.py --outdir ./out/getting_started
```

Check `./out/getting_started` for PNGs and any generated model files.

3) CI-friendly runs and temporary outputs
----------------------------------------

All examples that create files accept an `--outdir` flag so CI runs can
write artifacts to a temporary path that is cleaned up or uploaded by the
pipeline. Example test-friendly pattern using `pytest`'s `tmp_path`:

```python
# inside a pytest test
from pathlib import Path
import subprocess

outdir: Path = tmp_path / "example_out"
outdir.mkdir()
subprocess.run(["python", "examples/getting_started.py", "--outdir", str(outdir)], check=True)
```

Use the fake-runner (see the Examples page) in CI so tests don't require
`ngspice` to be installed.

4) Where to go next
--------------------

- Examples: `docs/examples.md` — overview and per-example links.
- Monte Carlo: `docs/monte-carlo-example.md` — report format and flags.
- API: `docs/api-reference.md` — how mkdocstrings is configured and which
	modules we recommend including in the API documentation.

Troubleshooting
---------------

- If an example fails with simulator errors, re-run with `--real-run` to
	confirm against `ngspice` and inspect the generated `.cir` or `.sp` file in
	the `--outdir` directory.
- If your CI environment cannot install `ngspice`, ensure the test uses the
	fake-runner and that `--outdir` is set to a writable temporary directory.

# Getting started

Create a Circuit and build a netlist:

```python
from cat.core.circuit import Circuit
from cat.core.components import Vdc, Resistor, Capacitor
from cat.core.net import GND

c = Circuit("rc_lowpass")
V1 = Vdc("1", 5.0)
R1 = Resistor("1", "1k")
C1 = Capacitor("1", "100n")

c.add(V1, R1, C1)
c.connect(V1.ports[0], R1.ports[0])
c.connect(R1.ports[1], C1.ports[0])
c.connect(V1.ports[1], GND)
c.connect(C1.ports[1], GND)

print(c.build_netlist())
```

This will print a SPICE netlist ready to be executed by a SPICE runner.

## Run a transient in one line

You can run a transient (.TRAN) analysis and get a pandas DataFrame directly:

```python
from cat.analysis import run_tran

df = run_tran(c, "10us", "5ms", return_df=True)
print(df.head())
```

Or keep the full result object for advanced usage:

```python
res = run_tran(c, "10us", "5ms")
print(res.traces.names)
```

Quickstart script
------------------

This repository includes a runnable `examples/getting_started.py` script that
walks through building a simple RC circuit and running AC, DC and transient
analyses, producing PNG plots. To run the tutorial:

```bash
cd examples
uv run --active python getting_started.py
```
