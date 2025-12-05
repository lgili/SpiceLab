"""Automated report generation for tolerance analysis results.

Provides functions to generate professional HTML reports from Monte Carlo
and WCA results, including histograms, process capability metrics, and
worst-case analysis tables.

Usage:
    from spicelab.analysis import monte_carlo, generate_report

    mc_result = monte_carlo(circuit, mapping, n=1000, ...)

    generate_report(
        mc_result,
        metric=lambda r: r.traces['V(vout)'].values[-1],
        lsl=2.4, usl=2.6,
        output_path="mc_report.html",
        title="Monte Carlo Analysis Report",
    )
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from .montecarlo import MonteCarloResult
from .result import AnalysisResult
from .stats import compute_stats
from .wca import WcaResult


@dataclass
class ReportSection:
    """A section in the report."""

    title: str
    """Section title."""

    content: str
    """HTML content for this section."""

    order: int = 0
    """Order in which sections appear (lower = earlier)."""


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    title: str = "Tolerance Analysis Report"
    """Report title."""

    show_histogram: bool = True
    """Whether to include histogram plot."""

    show_statistics: bool = True
    """Whether to include statistics table."""

    show_cpk: bool = True
    """Whether to include Cpk/yield metrics (requires lsl/usl)."""

    show_parameters: bool = True
    """Whether to include parameter variation table."""

    histogram_bins: int = 50
    """Number of bins for histogram."""

    include_css: bool = True
    """Whether to include inline CSS styling."""

    date_format: str = "%Y-%m-%d %H:%M:%S"
    """Date format for report timestamp."""


def _generate_histogram_base64(
    values: Sequence[float],
    lsl: float | None = None,
    usl: float | None = None,
    bins: int = 50,
    title: str = "Distribution",
    xlabel: str = "Value",
) -> str:
    """Generate a histogram as a base64-encoded PNG image.

    Returns empty string if matplotlib is not available.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram
    n, bins_edges, patches = ax.hist(
        values, bins=bins, density=True, alpha=0.7, color="steelblue", edgecolor="white"
    )

    # Add spec limits if provided
    if lsl is not None:
        ax.axvline(lsl, color="red", linestyle="--", linewidth=2, label=f"LSL = {lsl:.4g}")
    if usl is not None:
        ax.axvline(usl, color="red", linestyle="--", linewidth=2, label=f"USL = {usl:.4g}")

    # Add mean and sigma lines
    mean = sum(values) / len(values)
    std = (
        (sum((x - mean) ** 2 for x in values) / (len(values) - 1)) ** 0.5 if len(values) > 1 else 0
    )
    ax.axvline(mean, color="green", linestyle="-", linewidth=2, label=f"Mean = {mean:.4g}")
    ax.axvline(
        mean - 3 * std,
        color="orange",
        linestyle=":",
        linewidth=1.5,
        label=f"-3σ = {mean - 3*std:.4g}",
    )
    ax.axvline(
        mean + 3 * std,
        color="orange",
        linestyle=":",
        linewidth=1.5,
        label=f"+3σ = {mean + 3*std:.4g}",
    )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Save to base64
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return img_base64


def _format_number(value: float, precision: int = 4) -> str:
    """Format a number for display."""
    if abs(value) < 0.0001 or abs(value) >= 10000:
        return f"{value:.{precision}e}"
    return f"{value:.{precision}f}"


def _generate_css() -> str:
    """Generate CSS styles for the report."""
    return """
<style>
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
        background-color: #f5f5f5;
        color: #333;
    }
    .report-container {
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        padding: 30px;
        margin-bottom: 20px;
    }
    h1 {
        color: #2c3e50;
        border-bottom: 3px solid #3498db;
        padding-bottom: 10px;
    }
    h2 {
        color: #34495e;
        border-bottom: 1px solid #bdc3c7;
        padding-bottom: 8px;
        margin-top: 30px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
    }
    th, td {
        padding: 12px 15px;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }
    th {
        background-color: #3498db;
        color: white;
        font-weight: 600;
    }
    tr:hover {
        background-color: #f5f5f5;
    }
    .metric-value {
        font-family: 'Courier New', monospace;
        font-weight: bold;
    }
    .pass {
        color: #27ae60;
        font-weight: bold;
    }
    .fail {
        color: #e74c3c;
        font-weight: bold;
    }
    .warning {
        color: #f39c12;
        font-weight: bold;
    }
    .histogram-container {
        text-align: center;
        margin: 20px 0;
    }
    .histogram-container img {
        max-width: 100%;
        border-radius: 4px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .summary-box {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
        margin: 20px 0;
    }
    .summary-card {
        background-color: #ecf0f1;
        border-radius: 8px;
        padding: 20px;
        flex: 1;
        min-width: 200px;
        text-align: center;
    }
    .summary-card h3 {
        margin: 0 0 10px 0;
        color: #7f8c8d;
        font-size: 14px;
        text-transform: uppercase;
    }
    .summary-card .value {
        font-size: 28px;
        font-weight: bold;
        color: #2c3e50;
    }
    .footer {
        text-align: center;
        color: #95a5a6;
        font-size: 12px;
        margin-top: 30px;
        padding-top: 20px;
        border-top: 1px solid #ecf0f1;
    }
</style>
"""


def generate_monte_carlo_report(
    result: MonteCarloResult,
    metric: Callable[[AnalysisResult], float],
    lsl: float | None = None,
    usl: float | None = None,
    *,
    output_path: str | Path | None = None,
    title: str = "Monte Carlo Analysis Report",
    metric_name: str = "Output Metric",
    config: ReportConfig | None = None,
) -> str:
    """Generate an HTML report for Monte Carlo analysis results.

    Args:
        result: MonteCarloResult from monte_carlo() function.
        metric: Function to extract metric value from each run.
        lsl: Lower specification limit (optional).
        usl: Upper specification limit (optional).
        output_path: Path to save the HTML file. If None, returns HTML string.
        title: Report title.
        metric_name: Name of the metric being analyzed.
        config: Report configuration options.

    Returns:
        HTML string of the report.

    Example:
        html = generate_monte_carlo_report(
            mc_result,
            metric=lambda r: r.traces['V(vout)'].values[-1],
            lsl=2.4, usl=2.6,
            output_path="report.html",
            title="Vout Distribution Analysis",
            metric_name="Output Voltage (V)",
        )
    """
    cfg = config or ReportConfig(title=title)

    values = [metric(run) for run in result.runs]
    stats = compute_stats(values)

    # Start building HTML
    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='UTF-8'>",
        f"<title>{cfg.title}</title>",
    ]

    if cfg.include_css:
        html_parts.append(_generate_css())

    html_parts.extend(
        ["</head>", "<body>", "<div class='report-container'>", f"<h1>{cfg.title}</h1>"]
    )

    # Summary cards
    html_parts.append("<div class='summary-box'>")
    html_parts.append(f"""
        <div class='summary-card'>
            <h3>Samples</h3>
            <div class='value'>{stats.n:,}</div>
        </div>
        <div class='summary-card'>
            <h3>Mean</h3>
            <div class='value'>{_format_number(stats.mean)}</div>
        </div>
        <div class='summary-card'>
            <h3>Std Dev</h3>
            <div class='value'>{_format_number(stats.std)}</div>
        </div>
    """)

    if lsl is not None and usl is not None:
        cpk = stats.cpk(lsl, usl)
        yield_est = stats.yield_estimate(lsl, usl)
        failures = sum(1 for v in values if v < lsl or v > usl)
        actual_yield = (len(values) - failures) / len(values)

        cpk_class = "pass" if cpk >= 1.33 else ("warning" if cpk >= 1.0 else "fail")
        yield_class = (
            "pass" if actual_yield >= 0.99 else ("warning" if actual_yield >= 0.95 else "fail")
        )

        html_parts.append(f"""
            <div class='summary-card'>
                <h3>Cpk</h3>
                <div class='value {cpk_class}'>{cpk:.2f}</div>
            </div>
            <div class='summary-card'>
                <h3>Yield</h3>
                <div class='value {yield_class}'>{actual_yield * 100:.2f}%</div>
            </div>
        """)

    html_parts.append("</div>")  # End summary-box

    # Histogram
    if cfg.show_histogram:
        html_parts.append("<h2>Distribution Histogram</h2>")
        hist_base64 = _generate_histogram_base64(
            values,
            lsl,
            usl,
            bins=cfg.histogram_bins,
            title=f"{metric_name} Distribution (n={stats.n:,})",
            xlabel=metric_name,
        )
        if hist_base64:
            html_parts.append(f"""
                <div class='histogram-container'>
                    <img src='data:image/png;base64,{hist_base64}' alt='Histogram'>
                </div>
            """)
        else:
            html_parts.append("<p><em>Histogram not available (matplotlib not installed)</em></p>")

    # Statistics table
    if cfg.show_statistics:
        html_parts.append("<h2>Statistical Summary</h2>")
        html_parts.append("""
            <table>
                <tr><th>Statistic</th><th>Value</th></tr>
        """)
        stats_rows = [
            ("Sample Count", f"{stats.n:,}"),
            ("Mean", _format_number(stats.mean)),
            ("Standard Deviation", _format_number(stats.std)),
            ("Minimum", _format_number(stats.min)),
            ("Maximum", _format_number(stats.max)),
            ("Range", _format_number(stats.max - stats.min)),
            ("Median (P50)", _format_number(stats.median)),
            ("1st Percentile (P1)", _format_number(stats.p1)),
            ("5th Percentile (P5)", _format_number(stats.p5)),
            ("95th Percentile (P95)", _format_number(stats.p95)),
            ("99th Percentile (P99)", _format_number(stats.p99)),
            ("3-Sigma Low", _format_number(stats.sigma3_low)),
            ("3-Sigma High", _format_number(stats.sigma3_high)),
        ]
        for label, value in stats_rows:
            html_parts.append(f"<tr><td>{label}</td><td class='metric-value'>{value}</td></tr>")
        html_parts.append("</table>")

    # Process capability metrics
    if cfg.show_cpk and lsl is not None and usl is not None:
        html_parts.append("<h2>Process Capability Analysis</h2>")

        cpk = stats.cpk(lsl, usl)
        yield_est = stats.yield_estimate(lsl, usl)
        sigma_level = cpk * 3.0
        failures = sum(1 for v in values if v < lsl or v > usl)
        actual_yield = (len(values) - failures) / len(values)

        html_parts.append("""
            <table>
                <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
        """)

        # LSL/USL
        lsl_fmt = _format_number(lsl)
        usl_fmt = _format_number(usl)
        range_fmt = _format_number(usl - lsl)
        html_parts.append(f"""
            <tr><td>Lower Specification Limit (LSL)</td>
                <td class='metric-value'>{lsl_fmt}</td><td>-</td></tr>
            <tr><td>Upper Specification Limit (USL)</td>
                <td class='metric-value'>{usl_fmt}</td><td>-</td></tr>
            <tr><td>Specification Range</td>
                <td class='metric-value'>{range_fmt}</td><td>-</td></tr>
        """)

        # Cpk with status
        cpk_status = "pass" if cpk >= 1.33 else ("warning" if cpk >= 1.0 else "fail")
        cpk_text = (
            "Excellent"
            if cpk >= 2.0
            else (
                "Good"
                if cpk >= 1.67
                else ("Acceptable" if cpk >= 1.33 else ("Marginal" if cpk >= 1.0 else "Poor"))
            )
        )
        cpk_fmt = f"{cpk:.3f}"
        sigma_fmt = f"{sigma_level:.2f}σ"
        html_parts.append(f"""
            <tr><td>Process Capability (Cpk)</td>
                <td class='metric-value'>{cpk_fmt}</td>
                <td class='{cpk_status}'>{cpk_text}</td></tr>
            <tr><td>Sigma Level</td>
                <td class='metric-value'>{sigma_fmt}</td><td>-</td></tr>
        """)

        # Yield metrics
        yield_status = (
            "pass" if actual_yield >= 0.99 else ("warning" if actual_yield >= 0.95 else "fail")
        )
        yield_est_fmt = f"{yield_est * 100:.4f}%"
        actual_yield_fmt = f"{actual_yield * 100:.4f}%"
        fail_status = "pass" if failures == 0 else "fail"
        html_parts.append(f"""
            <tr><td>Estimated Yield (Normal)</td>
                <td class='metric-value'>{yield_est_fmt}</td><td>-</td></tr>
            <tr><td>Actual Yield (Empirical)</td>
                <td class='metric-value'>{actual_yield_fmt}</td>
                <td class='{yield_status}'>-</td></tr>
            <tr><td>Failures</td>
                <td class='metric-value'>{failures}</td>
                <td class='{fail_status}'>-</td></tr>
        """)

        html_parts.append("</table>")

    # Parameter table
    if cfg.show_parameters and result.mapping_manifest:
        html_parts.append("<h2>Varied Parameters</h2>")
        html_parts.append("""
            <table>
                <tr><th>Parameter</th><th>Nominal</th><th>Distribution</th></tr>
        """)
        for label, nominal, dist in result.mapping_manifest:
            html_parts.append(f"""
                <tr>
                    <td>{label}</td>
                    <td class='metric-value'>{_format_number(nominal)}</td>
                    <td>{dist}</td>
                </tr>
            """)
        html_parts.append("</table>")

    # Footer
    timestamp = datetime.now().strftime(cfg.date_format)
    html_parts.append(f"""
        <div class='footer'>
            <p>Generated by SpiceLab on {timestamp}</p>
            <p>Monte Carlo Analysis with {stats.n:,} iterations</p>
        </div>
    """)

    html_parts.extend(["</div>", "</body>", "</html>"])

    html_content = "\n".join(html_parts)

    # Save to file if path provided
    if output_path is not None:
        Path(output_path).write_text(html_content, encoding="utf-8")

    return html_content


def generate_wca_report(
    result: WcaResult,
    metric: Callable[[Any], float],
    *,
    output_path: str | Path | None = None,
    title: str = "Worst-Case Analysis Report",
    metric_name: str = "Output Metric",
    lsl: float | None = None,
    usl: float | None = None,
    config: ReportConfig | None = None,
) -> str:
    """Generate an HTML report for Worst-Case Analysis results.

    Args:
        result: WcaResult from run_wca() function.
        metric: Function to extract metric value from each corner.
        output_path: Path to save the HTML file. If None, returns HTML string.
        title: Report title.
        metric_name: Name of the metric being analyzed.
        lsl: Lower specification limit (optional).
        usl: Upper specification limit (optional).
        config: Report configuration options.

    Returns:
        HTML string of the report.
    """
    cfg = config or ReportConfig(title=title)

    # Get all corner values
    corner_values = result.all_values(metric)
    min_corner, min_val = corner_values[0]
    max_corner, max_val = corner_values[-1]

    # Start building HTML
    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='UTF-8'>",
        f"<title>{cfg.title}</title>",
    ]

    if cfg.include_css:
        html_parts.append(_generate_css())

    html_parts.extend(
        ["</head>", "<body>", "<div class='report-container'>", f"<h1>{cfg.title}</h1>"]
    )

    # Summary cards
    html_parts.append("<div class='summary-box'>")
    html_parts.append(f"""
        <div class='summary-card'>
            <h3>Corners Evaluated</h3>
            <div class='value'>{len(result.corners)}</div>
        </div>
        <div class='summary-card'>
            <h3>Minimum</h3>
            <div class='value'>{_format_number(min_val)}</div>
        </div>
        <div class='summary-card'>
            <h3>Maximum</h3>
            <div class='value'>{_format_number(max_val)}</div>
        </div>
        <div class='summary-card'>
            <h3>Range</h3>
            <div class='value'>{_format_number(max_val - min_val)}</div>
        </div>
    """)
    html_parts.append("</div>")

    # Worst-case corners table
    html_parts.append("<h2>Worst-Case Corners</h2>")
    html_parts.append("""
        <table>
            <tr><th>Rank</th><th>Corner</th><th>Metric Value</th><th>Status</th></tr>
    """)

    for rank, (corner, value) in enumerate(corner_values, 1):
        if lsl is not None and value < lsl:
            status = "<span class='fail'>Below LSL</span>"
        elif usl is not None and value > usl:
            status = "<span class='fail'>Above USL</span>"
        else:
            status = "<span class='pass'>In Spec</span>"

        html_parts.append(f"""
            <tr>
                <td>{rank}</td>
                <td>{corner.corner_name}</td>
                <td class='metric-value'>{_format_number(value)}</td>
                <td>{status}</td>
            </tr>
        """)

    html_parts.append("</table>")

    # Nominal values table
    if result.nominal_combo:
        html_parts.append("<h2>Component Nominal Values</h2>")
        html_parts.append("""
            <table>
                <tr><th>Component</th><th>Nominal Value</th><th>Tolerance</th></tr>
        """)
        for ref, nominal in result.nominal_combo.items():
            tol = result.tolerances.get(ref, 0)
            label = result.component_labels.get(ref, ref)
            html_parts.append(f"""
                <tr>
                    <td>{label}</td>
                    <td class='metric-value'>{_format_number(nominal)}</td>
                    <td>{tol * 100:.2f}%</td>
                </tr>
            """)
        html_parts.append("</table>")

    # Footer
    timestamp = datetime.now().strftime(cfg.date_format)
    html_parts.append(f"""
        <div class='footer'>
            <p>Generated by SpiceLab on {timestamp}</p>
            <p>Worst-Case Analysis with {len(result.corners)} corners</p>
        </div>
    """)

    html_parts.extend(["</div>", "</body>", "</html>"])

    html_content = "\n".join(html_parts)

    if output_path is not None:
        Path(output_path).write_text(html_content, encoding="utf-8")

    return html_content


# Convenience alias
generate_report = generate_monte_carlo_report


__all__ = [
    "ReportSection",
    "ReportConfig",
    "generate_monte_carlo_report",
    "generate_wca_report",
    "generate_report",
]
