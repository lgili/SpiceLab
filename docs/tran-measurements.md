# Transient measurements

Time-domain measurements available in `spicelab.analysis.measure` and tips to get robust results.

## Available specs

- RiseTimeSpec — 10–90% rise time between threshold crossings (configurable via `low_pct`/`high_pct`).
- THDSpec — Total Harmonic Distortion (%) from a steady-state sine; detects fundamental or accept `f0`.
- ENOBSpec — Effective number of bits from time-domain sine fit and SINAD: ENOB = (SINAD_dB − 1.76)/6.02.

## Usage

```python
from spicelab.analysis.measure import measure, RiseTimeSpec, THDSpec, ENOBSpec
rows = measure(ds, [
  RiseTimeSpec(name="tr", signal="V(out)"),
  THDSpec(name="thd", signal="V(out)", harmonics=5, f0=1e3),
  ENOBSpec(name="enob", signal="V(out)", harmonics=5, f0=1e3),
], return_as="python")
```

## Tips for accuracy

- Windowing and coherence: Internally, FFT helpers use Hann windows and a coherent RFFT when needed. For sine-based THD/ENOB, provide `f0` when possible and run for an integer number of cycles to reduce leakage.
- Sampling: Use sufficiently small `tstep` and simulate long enough to include multiple cycles after any initial transient. Discard startup cycles if ringing is present.
- ENOB method: The implementation fits a sine in time-domain (least squares) to avoid FFT scaling pitfalls, improving robustness in the presence of modest jitter and DC offsets.
- Edge cases: Rise time uses linear interpolation for threshold crossings; ensure the waveform crosses the defined thresholds once.

## Engine example

See `examples/ngspice_tran_metrics.py` for a minimal RC rise-time and a sine-based THD/ENOB using ngspice.
