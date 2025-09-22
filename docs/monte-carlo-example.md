# Monte Carlo example

This page documents the `examples/monte_carlo_demo.py` script and the
options it accepts.

Quick usage
-----------

```bash
python examples/monte_carlo_demo.py --n 200 --outdir ./mc_report
```

CLI flags of interest
---------------------

- `--n` — number of Monte Carlo trials (default smaller number for demos).
- `--outdir` — where to write plots and the HTML report.
- `--real-run` — if set, the script will call `ngspice` instead of the
  built-in fake-runner.

Report contents
---------------

The script writes a small HTML file that contains:

- a table of summary statistics (mean, std, CV and percentiles)
- images with the waveform, histograms, and a parameter trace matrix

Use `--outdir` to collect all artifacts in one place for inspection or CI
upload.
