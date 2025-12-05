"""Tests for report generation module."""

from __future__ import annotations

from spicelab.analysis.report import (
    ReportConfig,
    _format_number,
    _generate_css,
    generate_monte_carlo_report,
)


class TestFormatNumber:
    """Tests for _format_number helper."""

    def test_small_number(self):
        """Small numbers should use scientific notation."""
        result = _format_number(0.00001)
        assert "e" in result.lower()

    def test_large_number(self):
        """Large numbers should use scientific notation."""
        result = _format_number(100000)
        assert "e" in result.lower()

    def test_normal_number(self):
        """Normal numbers should use decimal notation."""
        result = _format_number(1.234)
        assert "e" not in result.lower()
        assert "1.234" in result

    def test_precision(self):
        """Should respect precision parameter."""
        result = _format_number(1.23456789, precision=2)
        assert "1.23" in result


class TestGenerateCss:
    """Tests for CSS generation."""

    def test_css_contains_style_tag(self):
        """CSS output should contain style tag."""
        css = _generate_css()
        assert "<style>" in css
        assert "</style>" in css

    def test_css_contains_body_styles(self):
        """CSS should contain body styling."""
        css = _generate_css()
        assert "body" in css
        assert "font-family" in css

    def test_css_contains_table_styles(self):
        """CSS should contain table styling."""
        css = _generate_css()
        assert "table" in css
        assert "border-collapse" in css


class TestReportConfig:
    """Tests for ReportConfig dataclass."""

    def test_default_values(self):
        """ReportConfig should have sensible defaults."""
        cfg = ReportConfig()
        assert cfg.title == "Tolerance Analysis Report"
        assert cfg.show_histogram is True
        assert cfg.show_statistics is True
        assert cfg.histogram_bins == 50

    def test_custom_values(self):
        """ReportConfig should accept custom values."""
        cfg = ReportConfig(
            title="Custom Report",
            show_histogram=False,
            histogram_bins=100,
        )
        assert cfg.title == "Custom Report"
        assert cfg.show_histogram is False
        assert cfg.histogram_bins == 100


class TestMonteCarloReportStructure:
    """Tests for Monte Carlo report structure (without actual data)."""

    def test_report_contains_title(self):
        """Report should contain the specified title."""
        # Create minimal mock result
        from dataclasses import dataclass
        from typing import Any

        @dataclass
        class MockAnalysisResult:
            traces: dict[str, Any]

        @dataclass
        class MockMonteCarloResult:
            samples: list[dict[str, float]]
            runs: list[MockAnalysisResult]
            mapping_manifest: list[tuple[str, float, str]] | None
            handles: list | None
            job: Any

        # Create mock with simple values
        mock_runs = [MockAnalysisResult(traces={}) for _ in range(10)]
        mock_samples = [{} for _ in range(10)]
        mock_result = MockMonteCarloResult(
            samples=mock_samples,
            runs=mock_runs,
            mapping_manifest=[("R1", 1000.0, "NormalPct(0.01)")],
            handles=None,
            job=None,
        )

        # Generate report with simple metric
        values = [1.0 + 0.01 * i for i in range(10)]
        idx = [0]

        def metric(run):
            val = values[idx[0]]
            idx[0] += 1
            return val

        html = generate_monte_carlo_report(
            mock_result,
            metric=metric,
            title="Test Report",
        )

        assert "Test Report" in html
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html

    def test_report_contains_statistics_section(self):
        """Report should contain statistics section when enabled."""
        from dataclasses import dataclass
        from typing import Any

        @dataclass
        class MockAnalysisResult:
            traces: dict[str, Any]

        @dataclass
        class MockMonteCarloResult:
            samples: list[dict[str, float]]
            runs: list[MockAnalysisResult]
            mapping_manifest: list[tuple[str, float, str]] | None
            handles: list | None
            job: Any

        mock_runs = [MockAnalysisResult(traces={}) for _ in range(10)]
        mock_samples = [{} for _ in range(10)]
        mock_result = MockMonteCarloResult(
            samples=mock_samples,
            runs=mock_runs,
            mapping_manifest=None,
            handles=None,
            job=None,
        )

        values = list(range(10))
        idx = [0]

        def metric(run):
            val = values[idx[0]]
            idx[0] += 1
            return float(val)

        html = generate_monte_carlo_report(
            mock_result,
            metric=metric,
            config=ReportConfig(show_statistics=True),
        )

        assert "Statistical Summary" in html
        assert "Mean" in html
        assert "Standard Deviation" in html

    def test_report_contains_cpk_when_limits_provided(self):
        """Report should contain Cpk section when spec limits are provided."""
        from dataclasses import dataclass
        from typing import Any

        @dataclass
        class MockAnalysisResult:
            traces: dict[str, Any]

        @dataclass
        class MockMonteCarloResult:
            samples: list[dict[str, float]]
            runs: list[MockAnalysisResult]
            mapping_manifest: list[tuple[str, float, str]] | None
            handles: list | None
            job: Any

        mock_runs = [MockAnalysisResult(traces={}) for _ in range(100)]
        mock_samples = [{} for _ in range(100)]
        mock_result = MockMonteCarloResult(
            samples=mock_samples,
            runs=mock_runs,
            mapping_manifest=None,
            handles=None,
            job=None,
        )

        import random

        random.seed(42)
        values = [random.gauss(5.0, 0.1) for _ in range(100)]
        idx = [0]

        def metric(run):
            val = values[idx[0]]
            idx[0] += 1
            return val

        html = generate_monte_carlo_report(
            mock_result,
            metric=metric,
            lsl=4.5,
            usl=5.5,
        )

        assert "Process Capability" in html
        assert "Cpk" in html
        assert "Yield" in html
        assert "LSL" in html
        assert "USL" in html


class TestReportOutput:
    """Tests for report output functionality."""

    def test_returns_html_string(self):
        """generate_monte_carlo_report should return HTML string."""
        from dataclasses import dataclass
        from typing import Any

        @dataclass
        class MockAnalysisResult:
            traces: dict[str, Any]

        @dataclass
        class MockMonteCarloResult:
            samples: list[dict[str, float]]
            runs: list[MockAnalysisResult]
            mapping_manifest: list[tuple[str, float, str]] | None
            handles: list | None
            job: Any

        mock_runs = [MockAnalysisResult(traces={}) for _ in range(5)]
        mock_result = MockMonteCarloResult(
            samples=[{} for _ in range(5)],
            runs=mock_runs,
            mapping_manifest=None,
            handles=None,
            job=None,
        )

        idx = [0]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        def metric(run):
            val = values[idx[0]]
            idx[0] += 1
            return val

        result = generate_monte_carlo_report(mock_result, metric=metric)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "html" in result.lower()

    def test_saves_to_file(self, tmp_path):
        """generate_monte_carlo_report should save to file when path provided."""
        from dataclasses import dataclass
        from typing import Any

        @dataclass
        class MockAnalysisResult:
            traces: dict[str, Any]

        @dataclass
        class MockMonteCarloResult:
            samples: list[dict[str, float]]
            runs: list[MockAnalysisResult]
            mapping_manifest: list[tuple[str, float, str]] | None
            handles: list | None
            job: Any

        mock_runs = [MockAnalysisResult(traces={}) for _ in range(5)]
        mock_result = MockMonteCarloResult(
            samples=[{} for _ in range(5)],
            runs=mock_runs,
            mapping_manifest=None,
            handles=None,
            job=None,
        )

        idx = [0]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        def metric(run):
            val = values[idx[0]]
            idx[0] += 1
            return val

        output_path = tmp_path / "report.html"
        generate_monte_carlo_report(mock_result, metric=metric, output_path=output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "html" in content.lower()
