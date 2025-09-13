# Changelog

All notable changes to this project will be documented in this file.

Format: Keep sections concise and scannable. Use Added / Changed / Fixed / Docs / Tests
when it helps clarity. Dates in ISO (YYYY-MM-DD).

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

## [Unreleased]
- Add entries here for upcoming changes.

---

Guidelines: Use Conventional Commit‑style bullets where helpful. Keep PR titles
clear; release notes are generated from this file when tagging versions.
