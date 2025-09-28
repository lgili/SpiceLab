# Changelog

All notable changes to this project will be documented in this file.

Format: Keep sections concise and scannable. Use Added / Changed / Fixed / Docs / Tests
when it helps clarity. Dates in ISO (YYYY-MM-DD).

## [Unreleased]
### Added
- Core Contracts doc (`docs/core-contracts.md`) descrevendo `AnalysisSpec`, `SweepSpec`, `Probe`, hashing e metadados.
- `Probe` suporte em `run_simulation(..., probes=[])` e serialização em `ResultMeta`/`DatasetResultHandle.attrs()`.

### Changed
- Migração M1: `AnalysisSpec` e `SweepSpec` de dataclasses para modelos Pydantic v2 (validação + serialização). Construtor posicional preservado.

### Tests
- Novos testes de validação negativa de sweep e args, probes e construtor posicional.

### Docs
- README inclui nota de migração M1.

### Removed
- Legacy `spicelab/core/models.py` (dataclass-era duplicate) fully removed after Pydantic migration; import from `spicelab.core.types` instead.

## [0.2.0] - 2025-09-28

M7 release: measurement framework, CLI UX, and pipeline refinements.

### Added
- DSP helpers (`spicelab.analysis.signal`): windowing, coherent RFFT, amplitude spectrum, PSD with ENBW.
- AC specs: `PhaseMarginSpec`, `GainBandwidthSpec`, `GainMarginSpec` (preserves complex AC values).
- TRAN specs: `RiseTimeSpec` (10–90% with interpolation), `THDSpec`, `ENOBSpec` (time‑domain sine fit).
- CLI `spicelab-measure` (RAW → JSON/CSV) with flags: `--ac`, `--tran`, `--num`, `--den`, `--signal`, `--f0`, `--harmonics`, `--format {json,csv,schema}`, `--list-signals`, `--list-details`, `--warn-nonfinite`, `--fail-on-nonfinite`.
- `--format schema`: prints header only (no measurement execution), safe for CI.
- Orchestrator/pipeline: `run_and_measure`, `measure_job_result`; example `examples/orchestrator_ac_measurements.py` exports CSV.

### Changed
- Precision: log‑frequency interpolation for PM/GBW/GM with safe fallbacks; robust GM selection near −180°.
- Stable column ordering in CLI and pipeline: `param_*` first, then known measurement fields; sanitized headers (no CR/LF, whitespace → `_`).
- CLI accepts NGSpice/LTspice/Xyce via `load_dataset`.
- CLI handles runtime exceptions (e.g., ENOB on short signals) as per‑metric `NaN`, honoring `--warn-nonfinite`/`--fail-on-nonfinite`.

### Docs
- AC/TRAN measurement pages; `docs/cli-ci.md` updated with schema/ordering/sanitization rules and examples.

### Tests
- Edge cases: RiseTime (no crossing, oscillations), AC (no unity crossing; sparse grid).
- CLI smokes for `schema`/`list-details` across NG/LT/XY; unit tests for ordering/header sanitization.

## [0.1.0] - 2025-09-13

### Added
- Components Guide with runnable snippets (`docs/components.md`).
- Components Quickstart table/cheat‑sheet (`docs/components-quickstart.md`).
- Device Models guide for `.model` usage (diodes, switches, sensors) (`docs/device-models.md`).
- MUX enable‑ports guide (`docs/mux-enable-ports.md`) and example `examples/mux_enable_ports_demo.py`.
- Examples gallery script (`examples/components_gallery.py`).
- CSV export in PT1000 Monte Carlo example (`--csv`), plus readable param labels.
- CLI flags for PT1000 example to skip plots with low N and per‑plot toggles.
- New tests covering:
  - OP/AC/TRAN/DC analyses on small circuits.
  - Sources: `Iac`, `Ipulse`; controlled sources: `VCVS`, `VCCS`; diode with `.model`.
  - AnalogMux8 OP wiring; MUX ngspice integration (static `sel`).
  - Monte Carlo: sanity checks, seed reproducibility, TRAN RC monotonicity.
  - Step grid with workers and progress, preserving order.
  - Robust TRAN metrics fallbacks (ringing/envelopes/complex traces).
  - RAW parser (ASCII, error paths, binary detection) and TraceSet → DataFrame.
  - Circuit introspection (summary/to_dot) and utilities (`units`, `e_series`, `topologies`).

### Changed
- Auto‑generated node names avoid colliding with user names (now `net_{id}`).
- Treat any net named "0" as ground across deep copies.
- PT1000 stats use `ddof=0` when N=1 to avoid NumPy warnings.

### Fixed
- ngspice runner: avoid duplicated `.end` by stripping final `.end` from netlist before
  appending directives/control/end.
- `AnalogMux8` enable‑ports control: tie switch `cn` to GND so `V(cp)−V(cn)` follows
  the enable pin correctly.

### Docs
- Expanded examples page, added guides and quickstarts to MkDocs navigation.

### Tests
- Overall coverage raised to ~83% with ngspice installed; critical paths (core/components/step)
  are well covered.

---

Guidelines: Use Conventional Commit‑style bullets where helpful. Keep PR titles
clear; release notes are generated from this file when tagging versions.
