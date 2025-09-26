"""Lightweight HTML/Markdown report builder for simulation artefacts."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from ..viz.plotly import VizFigure

polars_mod: ModuleType | None
try:  # Optional dependency for nice table rendering
    polars_mod = importlib.import_module("polars")
except Exception:  # pragma: no cover - optional
    polars_mod = None

if polars_mod is not None:
    _POLARS_DATAFRAME_TYPE: tuple[type[Any], ...] = (cast(type[Any], polars_mod.DataFrame),)
    _HAS_POLARS = True
else:
    _POLARS_DATAFRAME_TYPE = ()
    _HAS_POLARS = False

xarray_mod: ModuleType | None
try:  # Optional dependency for dataset summaries
    xarray_mod = importlib.import_module("xarray")
except Exception:  # pragma: no cover - optional
    xarray_mod = None

xr = cast(Any, xarray_mod)
_HAS_XARRAY = xr is not None

__all__ = ["ReportBuilder", "ReportSection"]


@dataclass
class ReportBlock:
    kind: str
    payload: Any
    caption: str | None = None


@dataclass
class ReportSection:
    """Container grouping report blocks under a heading."""

    title: str
    blocks: list[ReportBlock] = field(default_factory=list)

    def add_markdown(self, text: str) -> None:
        self.blocks.append(ReportBlock("markdown", text))

    def add_html(self, html: str) -> None:
        self.blocks.append(ReportBlock("html", html))

    def add_table(self, table: Any, *, caption: str | None = None) -> None:
        self.blocks.append(ReportBlock("table", table, caption))

    def add_figure(self, figure: VizFigure | str, *, caption: str | None = None) -> None:
        self.blocks.append(ReportBlock("figure", figure, caption))


@dataclass
class ReportBuilder:
    """High-level helper that produces HTML/Markdown reports."""

    title: str
    subtitle: str | None = None
    sections: list[ReportSection] = field(default_factory=list)

    def add_section(self, title: str) -> ReportSection:
        section = ReportSection(title)
        self.sections.append(section)
        return section

    # ------------------------------------------------------------------
    def to_markdown(self) -> str:
        figure_ids = self._figure_id_map()
        lines: list[str] = [f"# {self.title}"]
        if self.subtitle:
            lines.append(f"_{self.subtitle}_")
            lines.append("")
        for section in self.sections:
            lines.append(f"## {section.title}")
            for block in section.blocks:
                if block.kind == "markdown":
                    lines.append(block.payload.strip())
                elif block.kind == "html":
                    lines.append(block.payload.strip())
                elif block.kind == "table":
                    headers, rows = _table_to_rows(block.payload)
                    if headers:
                        lines.append("| " + " | ".join(headers) + " |")
                        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                        for row in rows:
                            lines.append("| " + " | ".join(row) + " |")
                    if block.caption:
                        lines.append(f"*{block.caption}*")
                elif block.kind == "figure":
                    figure_id = figure_ids[id(block)]
                    caption = block.caption or "figure"
                    lines.append(f"[{caption}](figures/{figure_id}.html)")
                lines.append("")
        return "\n".join(lines).strip() + "\n"

    def to_html(self) -> str:
        figure_ids = self._figure_id_map()
        parts: list[str] = [
            '<html><head><meta charset="utf-8">',
            "<style>",
            _DEFAULT_STYLE,
            "</style></head><body>",
        ]
        parts.append(f"<h1>{_escape(self.title)}</h1>")
        if self.subtitle:
            parts.append(f'<p class="subtitle">{_escape(self.subtitle)}</p>')
        for section in self.sections:
            parts.append(f"<section><h2>{_escape(section.title)}</h2>")
            for block in section.blocks:
                if block.kind == "markdown":
                    parts.append(_markdown_to_html(block.payload))
                elif block.kind == "html":
                    parts.append(block.payload)
                elif block.kind == "table":
                    headers, rows = _table_to_rows(block.payload)
                    parts.append("<table>")
                    if headers:
                        parts.append(
                            "<thead><tr>"
                            + "".join(f"<th>{_escape(h)}</th>" for h in headers)
                            + "</tr></thead>"
                        )
                    parts.append("<tbody>")
                    for row in rows:
                        parts.append(
                            "<tr>" + "".join(f"<td>{_escape(cell)}</td>" for cell in row) + "</tr>"
                        )
                    parts.append("</tbody></table>")
                    if block.caption:
                        parts.append(f'<p class="caption">{_escape(block.caption)}</p>')
                elif block.kind == "figure":
                    figure_id = figure_ids[id(block)]
                    title_attr = _escape(block.caption or "figure")
                    iframe_html = (
                        "<figure>\n"
                        f'  <iframe src="figures/{figure_id}.html" '
                        f'title="{title_attr}" '
                        'loading="lazy" frameborder="0" allowfullscreen>'
                        "</iframe>"
                    )
                    parts.append(iframe_html)
                    if block.caption:
                        parts.append(f"  <figcaption>{_escape(block.caption)}</figcaption>")
                    parts.append("</figure>")
            parts.append("</section>")
        parts.append("</body></html>")
        return "".join(parts)

    def write(
        self,
        output_dir: str | Path,
        *,
        slug: str | None = None,
        formats: Sequence[str] = ("html", "md"),
        include_figures: bool = True,
    ) -> Mapping[str, Path]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = slug or _slug(self.title)
        outputs: dict[str, Path] = {}
        if include_figures:
            self._export_figures(out_dir)
        wanted = {fmt.lower() for fmt in formats}
        if "html" in wanted:
            html_path = out_dir / f"{base}.html"
            html_path.write_text(self.to_html(), encoding="utf-8")
            outputs["html"] = html_path
        if "md" in wanted:
            md_path = out_dir / f"{base}.md"
            md_path.write_text(self.to_markdown(), encoding="utf-8")
            outputs["md"] = md_path
        return outputs

    # ------------------------------------------------------------------
    def _figure_id_map(self) -> dict[int, str]:
        mapping: dict[int, str] = {}
        for section_index, section in enumerate(self.sections, start=1):
            figure_index = 1
            for block in section.blocks:
                if block.kind != "figure":
                    continue
                mapping[id(block)] = f"sec{section_index:02d}-fig{figure_index:02d}"
                figure_index += 1
        return mapping

    def _export_figures(self, out_dir: Path) -> None:
        figure_ids = self._figure_id_map()
        fig_dir = out_dir / "figures"
        for section in self.sections:
            for block in section.blocks:
                if block.kind != "figure":
                    continue
                figure_id = figure_ids[id(block)]
                path = fig_dir / f"{figure_id}.html"
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = block.payload
                if isinstance(payload, VizFigure):
                    payload.to_html(path, include_plotlyjs="cdn")
                elif isinstance(payload, str):
                    path.write_text(payload, encoding="utf-8")
                elif hasattr(payload, "figure") and hasattr(payload.figure, "to_html"):
                    html = payload.figure.to_html(full_html=True, include_plotlyjs="cdn")
                    path.write_text(html, encoding="utf-8")
                else:  # pragma: no cover - safety net
                    raise TypeError("Unsupported figure payload type")


def _table_to_rows(table: Any) -> tuple[list[str], list[list[str]]]:
    if _HAS_POLARS:
        try:
            if isinstance(table, _POLARS_DATAFRAME_TYPE):
                headers = list(table.columns)
                rows = [[_stringify(cell) for cell in row] for row in table.iter_rows()]
                return headers, rows
        except TypeError:
            pass

    if hasattr(table, "to_pandas"):
        pdf = table.to_pandas()
        headers = list(pdf.columns)
        rows = [[_stringify(cell) for cell in row] for row in pdf.to_numpy()]
        return headers, rows

    if isinstance(table, Sequence):
        seq = list(table)
        if not seq:
            return [], []
        first = seq[0]
        if isinstance(first, Mapping):
            headers = list(first.keys())
            rows = [[_stringify(row.get(col, "")) for col in headers] for row in seq]
            header_labels = [_stringify(col) for col in headers]
            return header_labels, rows
        headers = [f"col{i + 1}" for i in range(len(first))]
        rows = [[_stringify(cell) for cell in row] for row in seq]
        return headers, rows

    if hasattr(table, "items"):
        items = list(table.items())
        headers = ["key", "value"]
        rows = [[_stringify(k), _stringify(v)] for k, v in items]
        return headers, rows

    raise TypeError("Unsupported table type for reporting")


def dataset_summary(ds: Any) -> tuple[list[str], list[list[str]]]:
    if _HAS_XARRAY and xr is not None and isinstance(ds, xr.Dataset):
        rows: list[list[str]] = []
        for dim, size in ds.sizes.items():
            rows.append(["dimension", _stringify(dim), str(size)])
        for name, data in ds.data_vars.items():
            shape = " × ".join(str(value) for value in data.shape)
            rows.append(["variable", _stringify(name), shape or "scalar"])
        return ["kind", "name", "details"], rows
    raise TypeError("dataset_summary expects an xarray.Dataset when available")


def _markdown_to_html(text: str) -> str:
    paragraphs = [f"<p>{_escape(p.strip())}</p>" for p in text.split("\n\n") if p.strip()]
    return "".join(paragraphs)


def _slug(text: str) -> str:
    safe = [c.lower() if c.isalnum() else "-" for c in text.strip()]
    slug = "".join(safe).strip("-")
    return slug or "section"


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


_DEFAULT_STYLE = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; }
h1 { margin-bottom: 0.25rem; }
.subtitle { color: #555; margin-top: 0; }
section { margin-bottom: 2rem; }
table { border-collapse: collapse; margin: 1rem 0; width: 100%; }
th, td { border: 1px solid #ccc; padding: 0.5rem; text-align: left; }
figcaption { font-size: 0.9rem; color: #555; }
iframe { width: 100%; min-height: 320px; border: none; }
.caption { font-style: italic; }
"""
