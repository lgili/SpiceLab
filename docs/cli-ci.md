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
