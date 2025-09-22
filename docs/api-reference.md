# API reference guidance

This project uses `mkdocstrings` to collect API documentation from the
`src/` package. The handler is configured in `mkdocs.yml` with a `search_path`
that points to `src` so local imports resolve during the site build.

How to add a module to the API reference
----------------------------------------

1. Add a markdown file under `docs/reference/` and include the `::: path.to.module`
   directive where you want the auto-generated docstring content to appear.

Example: `docs/reference/analysis.md`

```markdown
::: cat.analysis
```

2. Run `mkdocs build` locally to generate the site and verify that
   mkdocstrings finds the module. If mkdocstrings fails to import the module,
   ensure your virtualenv has the package installed (`pip install -e .`) and
   that `mkdocs` is executed from the repository root.

3. Keep exported API stable: prefer documenting public top-level methods and
   classes from the `cat.*` namespace. Avoid heavy implementation details; use
   examples for usage patterns.

Troubleshooting imports
-----------------------

- If mkdocstrings complains about missing dependencies, install them into the
  environment used for the docs build (dev-requirements.txt).
- If you need to alter the import path, update `mkdocs.yml` `plugins.mkdocstrings.python.extra.search_path`.

Packaging tips
--------------

- Keep `pyproject.toml` up to date with any extra dependencies needed for
  examples you intend to show in the API docs. This keeps local builds
  reproducible.
- When publishing the site from CI, make sure the build job creates an
  environment that includes dev/test dependencies so mkdocstrings can import
  modules without errors.
