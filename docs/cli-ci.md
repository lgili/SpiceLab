# CLI and CI guide

# spicelab scripts are regular Python modules. Most of them print summary
information and optionally write PNG/CSV files next to the script. None of them
require custom CLI flags any more — use environment variables (`SPICELAB_ENGINE`)
if you want to force a particular simulator.

## Running examples in CI

- Execute scripts with `uv run --active python examples/<script>.py`. Use a
  temporary working directory when you want to capture artefacts.
- Cache directories created by the orchestrator (default `.spicelab_cache`) can
  be retained between CI runs to avoid re-simulating identical sweeps.
- If you need to guarantee that a script does not invoke a real simulator,
  temporarily patch `spicelab.engines.factory.create_simulator` with a stub, as
  shown in the unit tests (`tests/test_engine_multi_parity.py`).

## Example pytest snippet

```python
import os
import subprocess


def test_rc_tran_smoke(tmp_path):
    env = {"SPICELAB_ENGINE": "ngspice", **os.environ}
    subprocess.run(
        ["uv", "run", "--active", "python", "examples/rc_tran.py"],
        cwd=tmp_path,
        env=env,
        check=True,
    )
```

Swap `examples/rc_tran.py` for any other script you need to exercise.

## spicelab-measure (RAW -> JSON/CSV)

Use the built-in CLI to compute AC and/or TRAN measurements from an NGSpice `.raw` file and export JSON or CSV.

Examples:

```bash
# AC + TRAN, JSON on stdout
spicelab-measure path/to/ac_or_tran.raw --ac --tran --num "V(out)" --den "V(in)" --signal "V(out)" --f0 1e3 --format json

# TRAN only, CSV to file
spicelab-measure path/to/tran.raw --tran --signal "V(out)" --f0 1e3 --format csv --out metrics.csv
```

Columns include the measurement fields and values; JSON uses a list of dicts compatible with Python consumption.

Tips:

- Discover available variables with:

  ```bash
  spicelab-measure path/to/file.raw --list-signals --format csv
  ```

- Include dtype and axis when listing signals:

  ```bash
  spicelab-measure path/to/file.raw --list-signals --list-details --format csv
  ```

- Print just the CSV schema (columns) for the selected metrics:

  ```bash
  spicelab-measure path/to/file.raw --ac --format schema --num "V(out)" --den "V(in)"
  ```

If a provided `--num`, `--den`, or `--signal` is invalid, the CLI shows suggestions based on available names.

Schema and column ordering:

- `--format schema` prints only the header (CSV columns) and does not execute any measurement logic. This makes it safe for CI checks or planning downstream pipelines.
- Column order is stable and sanitized:
  - Core fields first: `measure,type,value,units`
  - Then known fields depending on selected metrics, e.g. AC: `freq,numerator,denominator,tolerance_deg`; TRAN: `signal,harmonics,f0,low_pct,high_pct,sinad_db`.
  - Remaining fields are sorted by sanitized name for stability.
  - Header names are sanitized (whitespace/newlines collapsed to `_` and CR/LF stripped) to avoid CSV header artifacts from unusual RAW keys.

Quick checks:

```bash
# AC schema (using fixture paths in this repo)
uv run --active python -m spicelab.cli.measure tests/fixtures/rc_ac_ng.raw --ac --num "V(out)" --den "I(R1)" --format schema

# List signals with details
uv run --active python -m spicelab.cli.measure tests/fixtures/rc_ac_ng.raw --list-signals --list-details --format csv
```

## Orchestrator + measurements pipeline

You can run sweeps with the orchestrator and aggregate measurements per run:

```python
from spicelab.analysis import run_and_measure, PhaseMarginSpec, GainBandwidthSpec, GainMarginSpec
from spicelab.orchestrator import Job
from spicelab.core.types import AnalysisSpec, SweepSpec

job = Job(
  circuit=your_circuit,
  analyses=[AnalysisSpec("ac", {"sweep_type": "dec", "n": 20, "fstart": 10, "fstop": 1e6})],
  sweep=SweepSpec(variables={"R1": [1e3, 2e3, 4e3]}),
)

rows = run_and_measure(
  job,
  [
    PhaseMarginSpec(name="pm", numerator="V(out)", denominator="V(in)"),
    GainBandwidthSpec(name="gbw", numerator="V(out)", denominator="V(in)"),
    GainMarginSpec(name="gm", numerator="V(out)", denominator="V(in)"),
  ],
  return_as="python",
)
```

Each row includes the measurement fields plus the sweep parameters (prefixed as `param_<name>` by default). For a Polars DataFrame instead, pass `return_as="polars"`.

CSV exports in examples/orchestrator scripts follow the same stable ordering rules used by the CLI:

- `param_*` columns come first (sorted).
- Measurement fields are ordered using the same helper as the CLI.
- Headers are sanitized to remove CR/LF and collapse whitespace to `_`.
