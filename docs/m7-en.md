# M7 — Measurements, DSP, and Control Metrics

This milestone delivered a reusable measurement toolkit (similar to SPICE `.meas`), signal processing helpers, and CLI/pipeline UX for CI-friendly exports.

Highlights:

- AC: Phase Margin, Gain Bandwidth, Gain Margin with log-frequency interpolation and safe fallbacks.
- TRAN: Rise Time (10–90% with interpolation), THD, ENOB (time-domain sine fit).
- DSP helpers: windowing, coherent RFFT, amplitude spectrum, PSD with ENBW.
- CLI `spicelab-measure`: JSON/CSV output, header-only schema, signal discovery with dtype/axis.
- Orchestrator pipeline: stable CSV ordering (`param_*` first) and header sanitization.

See the M7 notebooks under `notebooks/` for runnable examples.
