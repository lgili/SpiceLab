# AC Stability (Bode, PM/GM)

Compute Bode magnitude/phase and stability margins from an AC run.

![RC Bode](assets/examples/ac_bode.png)

Note: NGSpice names nodes as `n1`, `n2`, ... in the RAW file by default. Use `v(n1)`
for the first node voltage (output in the RC examples).

## Quick Bode

```python
from spicelab.analysis import bode

# Run an AC sweep and get (f, mag_db, phase_deg)
f, mag_db, ph = bode(c, y_out="v(n1)", sweep_type="dec", n=201, fstart=10.0, fstop=1e6)
```

The circuit must include a small-signal AC source (e.g., `VA()` or `Iac`).

## Margins and bandwidth

```python
from spicelab.analysis import ac_gain_phase, bandwidth_3db, crossover_freq_0db, phase_margin, gain_margin_db

res = run_ac(c, "dec", 201, 10.0, 1e6)
f, g_db, ph = ac_gain_phase(res.traces, y_out="v(n1)")

bw = bandwidth_3db(res.traces, y_out="v(n1)")
wc = crossover_freq_0db(res.traces, y_out="v(n1)")
pm = phase_margin(res.traces, y_out="v(n1)")
gm = gain_margin_db(res.traces, y_out="v(n1)")
print(bw, wc, pm, gm)
```

Use `plot_bode(ts, y)` to open an interactive Plotly Bode chart when a complex trace is available.
