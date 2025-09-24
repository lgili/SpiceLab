# API reference guidance

This project uses `mkdocstrings` to collect API documentation from the
`src/` package. The handler is configured in `mkdocs.yml` with a `search_path`
that points to `src` so local imports resolve during the site build.
This project uses `mkdocstrings` to collect API documentation from the
root package (`spicelab`). The handler is configured in `mkdocs.yml` and
searches the repository root so local imports resolve during the site build.

How to add a module to the API reference

1. Add a markdown file under `docs/reference/` and include the `::: path.to.module`
   directive where you want the auto-generated docstring content to appear.

Example: `docs/reference/analysis.md`

```markdown
::: spicelab.analysis
```

2. Run `mkdocs build` locally to generate the site and verify that
   mkdocstrings finds the module. If mkdocstrings fails to import the module,
   ensure your virtualenv has the package installed (`pip install -e .`) and
   that `mkdocs` is executed from the repository root.

3. Keep exported API stable: prefer documenting public top-level methods and
   classes from the `spicelab.*` namespace. Avoid heavy implementation details; use
   examples for usage patterns.

Troubleshooting imports

  environment used for the docs build (dev-requirements.txt).

Packaging tips

  examples you intend to show in the API docs. This keeps local builds
  reproducible.
  environment that includes dev/test dependencies so mkdocstrings can import
  modules without errors.
