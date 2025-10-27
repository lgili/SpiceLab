# M11 - UX Enhancements: Widgets, CLI, and Documentation

## Why
SpiceLab needs professional-grade user experience to compete with commercial tools. Users struggle with: engine installation/detection, interactive parameter exploration in notebooks, and learning curve for new features. Better UX (widgets, diagnostic tools, comprehensive docs) will dramatically improve adoption and productivity.

## What Changes
- Add `spicelab/widgets/` with interactive Jupyter widgets for parameter sweeps and plot exploration
- Implement `spicelab doctor` CLI command to diagnose installation and suggest fixes
- Add `spicelab convert` CLI for format conversion (RAW → Parquet, etc.)
- Create comprehensive MkDocs documentation with getting started, cookbook, API reference
- Improve error messages with actionable suggestions and context
- Add plotting helpers in `spicelab/viz/` for common visualizations (Bode, step, Nyquist)

## Impact
- **Affected specs**: widgets, cli-tools, documentation, error-handling, visualization
- **Affected code**:
  - New: `spicelab/widgets/`, `spicelab/cli.py`, `spicelab/viz/`
  - New: `docs/` (MkDocs structure)
  - Modified: All error messages across codebase for consistency
  - New: `examples/notebooks/` with interactive widget demos
- **Dependencies**: ipywidgets, plotly, mkdocs, mkdocs-material
