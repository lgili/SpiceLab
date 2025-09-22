# CLI and CI guide

This page documents the common CLI flags supported by examples and
recommended patterns for running examples in CI.

Common flags
------------

- `--outdir PATH` — write all generated artifacts (plots, model files,
  reports) to PATH. Examples default to the current working directory. For CI
  use a temporary directory (see pattern below).
- `--real-run` — when present, the example invokes the real simulator
  (`ngspice`) instead of the internal fake-runner used for fast CI runs.
- `--n` or `--trials` — number of Monte Carlo trials (used by
  `monte_carlo_demo.py`).

CI recommendations
------------------

1. Use a temporary output directory and pass it with `--outdir` so test
artifacts don't leak into the repo. Example pytest pattern:

```python
def test_example_smoke(tmp_path):
    outdir = tmp_path / "example_out"
    outdir.mkdir()
    subprocess.run(["python", "examples/monte_carlo_demo.py", "--outdir", str(outdir)], check=True)
    assert any(outdir.iterdir())
```

2. Prefer the fake-runner in automated tests. The fake-runner provides a
deterministic, fast output that is sufficient for smoke tests and
visualization pipelines.

3. Run heavier simulations with `--real-run` only in dedicated integration
tests or developer runs where `ngspice` is available.

4. Upload the `--outdir` contents as CI artifacts when you need to inspect
plots or the HTML report.

How to enable the fake-runner in tests
--------------------------------------

Examples are written to select the fake-runner by default when the real
simulator is unavailable. If you need to force the fake-runner from code,
most examples accept an entrypoint to pass the runner object directly. See
the example module docstrings for details.
