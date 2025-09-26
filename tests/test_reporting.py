from __future__ import annotations

from pathlib import Path

import pytest
from spicelab.reporting import ReportBuilder
from spicelab.reporting.report import dataset_summary


def test_report_builder_generates_markdown_and_html(tmp_path: Path) -> None:
    builder = ReportBuilder("RC Low-pass", subtitle="Demo run")
    summary = builder.add_section("Summary")
    summary.add_markdown("Circuit converged without warnings.")
    summary.add_table(
        [
            {"name": "R1", "value": "1k"},
            {"name": "C1", "value": "100n"},
        ],
        caption="Component values",
    )
    summary.add_figure("<div>fake figure</div>", caption="Response plot")

    outputs = builder.write(tmp_path, slug="rc-report")

    html_path = outputs["html"]
    md_path = outputs["md"]
    assert html_path.exists()
    assert md_path.exists()

    html = html_path.read_text(encoding="utf-8")
    md = md_path.read_text(encoding="utf-8")
    assert "Circuit converged" in html
    assert "Component values" in html
    assert "Circuit converged" in md
    assert "Component values" in md

    figure_files = list((tmp_path / "figures").glob("*.html"))
    assert len(figure_files) == 1
    assert "fake figure" in figure_files[0].read_text(encoding="utf-8")


def test_dataset_summary_builds_rows() -> None:
    xr = pytest.importorskip("xarray")

    ds = xr.Dataset({"V(out)": ("time", [0.0, 1.0, 2.0])}, coords={"time": [0.0, 0.5, 1.0]})
    headers, rows = dataset_summary(ds)
    assert headers == ["kind", "name", "details"]
    kinds = {row[0] for row in rows}
    assert {"dimension", "variable"}.issubset(kinds)
