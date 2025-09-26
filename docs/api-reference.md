# API reference guidance

Documentation for public modules is generated with `mkdocstrings`. To expose new
API pages:

1. Add a Markdown file under `docs/reference/`.
2. Place a directive such as `::: spicelab.analysis` in the file.
3. Run `mkdocs build` to verify imports succeed (install the package in editable
   mode so mkdocstrings can import it).

Keep exports stable and prefer documenting the orchestrator and core component
APIs (`spicelab.analysis`, `spicelab.core`, `spicelab.engines`, `spicelab.io`).
