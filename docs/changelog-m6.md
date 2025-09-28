# M6 Wrap-up

This milestone focused on unifying the branding to "spicelab", improving notebook UX, and hardening typing and CI hooks.

Highlights:

- Branding: updated README, docs content, and site name to "spicelab".
- Notebooks: added six English tutorials covering setup, sweeps, Monte Carlo, viz/widgets, and the DSL builder.
- Notebook plotting: improved robustness with a lightweight FigureWidget shim when ipywidgets/plotly aren't fully available.
- Tooling: resolved pre-commit failures (ruff/mypy), added type hints, and updated config for optional stubs.
- Tests: targeted tests for viz wrappers; added smoke tests for circuit tables.

Next:

- Integrate notebooks more deeply (optional plugin) or continue linking directly.
- Start M7 scope.
