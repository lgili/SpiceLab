# AC Stability (Bode, PM/GM)

Compute Bode magnitude/phase and stability margins from an AC run.

![RC Bode](assets/examples/ac_bode.png)

## Quick Bode

```python
from cat.analysis import bode

# Run an AC sweep and get (f, mag_db, phase_deg)
f, mag_db, ph = bode(c, y_out="v(out)", sweep_type="dec", n=201, fstart=10.0, fstop=1e6)
```

The circuit must include a small-signal AC source (e.g., `VA()` or `Iac`).

## Margins and bandwidth

```python
from cat.analysis import ac_gain_phase, bandwidth_3db, crossover_freq_0db, phase_margin, gain_margin_db

res = run_ac(c, "dec", 201, 10.0, 1e6)
f, g_db, ph = ac_gain_phase(res.traces, y_out="v(out)")

bw = bandwidth_3db(res.traces, y_out="v(out)")
wc = crossover_freq_0db(res.traces, y_out="v(out)")
pm = phase_margin(res.traces, y_out="v(out)")
gm = gain_margin_db(res.traces, y_out="v(out)")
print(bw, wc, pm, gm)
```

Use `plot_bode(ts, y)` to plot the complex trace when available.
