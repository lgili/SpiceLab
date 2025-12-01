"""Report Generator Plugin.

This plugin automatically generates simulation reports in various formats
(Markdown, HTML, JSON) after simulations complete.

Usage::

    from spicelab.plugins import PluginManager
    from spicelab.plugins.examples import ReportGeneratorPlugin

    manager = PluginManager()
    plugin = manager.loader.load_from_class(ReportGeneratorPlugin)
    manager.registry.register(plugin)
    manager.activate_plugin("report-generator")

    # Configure output
    manager.set_plugin_settings("report-generator", {
        "output_dir": "./reports",
        "format": "markdown",  # or "html", "json", "all"
        "include_netlist": True,
        "include_plots": True,
    })

    # Run simulation - report is auto-generated
    result = circuit.run([AnalysisSpec("tran", {"tstep": "1u", "tstop": "1m"})])
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..base import Plugin, PluginMetadata, PluginType
from ..hooks import HookManager, HookPriority, HookType


class ReportGeneratorPlugin(Plugin):
    """Plugin that auto-generates simulation reports.

    Features:
    - Auto-generates reports after each simulation
    - Supports Markdown, HTML, and JSON formats
    - Includes circuit info, analysis results, and optionally netlists
    - Configurable output directory and format
    """

    def __init__(self) -> None:
        self._config: dict[str, Any] = {
            "output_dir": "./reports",
            "format": "markdown",
            "include_netlist": True,
            "include_raw_data": False,
            "auto_open": False,
            "template": "default",
        }
        self._current_circuit: Any = None
        self._current_analyses: list[Any] = []
        self._simulation_start: datetime | None = None

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="report-generator",
            version="1.0.0",
            description="Auto-generate simulation reports in Markdown/HTML/JSON",
            author="SpiceLab Team",
            plugin_type=PluginType.GENERIC,
            keywords=["report", "documentation", "export"],
        )

    def configure(self, settings: dict[str, Any]) -> None:
        """Configure the report generator.

        Args:
            settings: Configuration with keys:
                - output_dir: Directory for reports
                - format: "markdown", "html", "json", or "all"
                - include_netlist: Include netlist in report
                - include_raw_data: Include raw simulation data
                - auto_open: Open report after generation
        """
        self._config.update(settings)
        # Ensure output directory exists
        Path(self._config["output_dir"]).mkdir(parents=True, exist_ok=True)

    def activate(self) -> None:
        """Activate the plugin and register hooks."""
        Path(self._config["output_dir"]).mkdir(parents=True, exist_ok=True)
        self._register_hooks()

    def deactivate(self) -> None:
        """Deactivate the plugin."""
        pass

    def _register_hooks(self) -> None:
        """Register hooks for report generation."""
        hook_manager = HookManager.get_instance()

        hook_manager.register_hook(
            HookType.PRE_SIMULATION,
            self._on_pre_simulation,
            priority=HookPriority.LOW,
            plugin_name=self.name,
            description="Capture simulation context",
        )

        hook_manager.register_hook(
            HookType.POST_SIMULATION,
            self._on_post_simulation,
            priority=HookPriority.LOWEST,
            plugin_name=self.name,
            description="Generate simulation report",
        )

    def _on_pre_simulation(self, **kwargs: Any) -> None:
        """Capture pre-simulation context."""
        self._current_circuit = kwargs.get("circuit")
        self._current_analyses = kwargs.get("analyses", [])
        self._simulation_start = datetime.now()

    def _on_post_simulation(self, **kwargs: Any) -> None:
        """Generate report after simulation."""
        result = kwargs.get("result")
        if not result:
            return

        duration = None
        if self._simulation_start:
            duration = (datetime.now() - self._simulation_start).total_seconds()

        report_data = self._build_report_data(result, duration)

        fmt = self._config["format"].lower()
        if fmt == "all":
            self._generate_markdown(report_data)
            self._generate_html(report_data)
            self._generate_json(report_data)
        elif fmt == "html":
            self._generate_html(report_data)
        elif fmt == "json":
            self._generate_json(report_data)
        else:
            self._generate_markdown(report_data)

    def _build_report_data(self, result: Any, duration: float | None) -> dict[str, Any]:
        """Build report data structure."""
        circuit_name = "unknown"
        component_count = 0
        netlist = ""

        if self._current_circuit:
            circuit_name = getattr(self._current_circuit, "name", "unknown")
            components = getattr(self._current_circuit, "_components", [])
            component_count = len(components)
            if self._config["include_netlist"]:
                try:
                    netlist = self._current_circuit.build_netlist()
                except Exception:
                    netlist = "Error building netlist"

        # Extract analysis results
        analyses_data = []
        for analysis in self._current_analyses:
            analysis_info = {
                "type": getattr(analysis, "mode", str(analysis)),
                "params": getattr(analysis, "params", {}),
            }
            analyses_data.append(analysis_info)

        # Extract result data
        result_data: dict[str, Any] = {}
        if hasattr(result, "traces"):
            result_data["traces"] = list(result.traces.keys()) if result.traces else []
        if hasattr(result, "data"):
            result_data["data_shape"] = (
                str(result.data.shape) if hasattr(result.data, "shape") else "N/A"
            )

        return {
            "timestamp": datetime.now().isoformat(),
            "circuit": {
                "name": circuit_name,
                "component_count": component_count,
                "netlist": netlist if self._config["include_netlist"] else None,
            },
            "simulation": {
                "duration_seconds": duration,
                "analyses": analyses_data,
            },
            "results": result_data,
        }

    def _get_report_path(self, extension: str) -> Path:
        """Get unique report file path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        circuit_name = "unknown"
        if self._current_circuit:
            circuit_name = getattr(self._current_circuit, "name", "unknown")
        filename = f"report_{circuit_name}_{timestamp}.{extension}"
        return Path(self._config["output_dir"]) / filename

    def _generate_markdown(self, data: dict[str, Any]) -> Path:
        """Generate Markdown report."""
        path = self._get_report_path("md")

        lines = [
            f"# Simulation Report: {data['circuit']['name']}",
            "",
            f"**Generated:** {data['timestamp']}",
            "",
            "## Circuit Information",
            "",
            f"- **Name:** {data['circuit']['name']}",
            f"- **Components:** {data['circuit']['component_count']}",
            "",
            "## Simulation",
            "",
        ]

        if data["simulation"]["duration_seconds"]:
            lines.append(
                f"- **Duration:** {data['simulation']['duration_seconds']:.3f}s"
            )

        lines.append("")
        lines.append("### Analyses")
        lines.append("")
        for analysis in data["simulation"]["analyses"]:
            lines.append(f"- **{analysis['type']}**: {analysis['params']}")

        lines.append("")
        lines.append("## Results")
        lines.append("")

        if data["results"].get("traces"):
            lines.append("### Available Traces")
            lines.append("")
            for trace in data["results"]["traces"]:
                lines.append(f"- `{trace}`")

        if data["circuit"].get("netlist"):
            lines.append("")
            lines.append("## Netlist")
            lines.append("")
            lines.append("```spice")
            lines.append(data["circuit"]["netlist"])
            lines.append("```")

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _generate_html(self, data: dict[str, Any]) -> Path:
        """Generate HTML report."""
        path = self._get_report_path("html")

        traces_html = ""
        if data["results"].get("traces"):
            traces_html = "<ul>" + "".join(
                f"<li><code>{t}</code></li>" for t in data["results"]["traces"]
            ) + "</ul>"

        netlist_html = ""
        if data["circuit"].get("netlist"):
            netlist_html = f"""
            <h2>Netlist</h2>
            <pre><code>{data['circuit']['netlist']}</code></pre>
            """

        duration_str = ""
        if data["simulation"]["duration_seconds"]:
            duration_str = f"{data['simulation']['duration_seconds']:.3f}s"

        analyses_html = "".join(
            f"<li><strong>{a['type']}</strong>: {a['params']}</li>"
            for a in data["simulation"]["analyses"]
        )

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Simulation Report: {data['circuit']['name']}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 900px; margin: 40px auto; padding: 20px; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        pre {{ background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        code {{ background: #e9ecef; padding: 2px 6px; border-radius: 3px; }}
        .meta {{ color: #7f8c8d; font-size: 0.9em; }}
        .info-grid {{ display: grid; grid-template-columns: 150px 1fr; gap: 10px; }}
        .info-grid dt {{ font-weight: bold; color: #555; }}
    </style>
</head>
<body>
    <h1>Simulation Report: {data['circuit']['name']}</h1>
    <p class="meta">Generated: {data['timestamp']}</p>

    <h2>Circuit Information</h2>
    <dl class="info-grid">
        <dt>Name</dt><dd>{data['circuit']['name']}</dd>
        <dt>Components</dt><dd>{data['circuit']['component_count']}</dd>
    </dl>

    <h2>Simulation</h2>
    <dl class="info-grid">
        <dt>Duration</dt><dd>{duration_str or 'N/A'}</dd>
    </dl>

    <h3>Analyses</h3>
    <ul>{analyses_html}</ul>

    <h2>Results</h2>
    <h3>Available Traces</h3>
    {traces_html or '<p>No traces available</p>'}

    {netlist_html}
</body>
</html>"""

        path.write_text(html, encoding="utf-8")
        return path

    def _generate_json(self, data: dict[str, Any]) -> Path:
        """Generate JSON report."""
        path = self._get_report_path("json")
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path

    # Public API for manual report generation

    def generate_report(
        self,
        circuit: Any,
        result: Any,
        *,
        format: str | None = None,
        output_path: str | Path | None = None,
    ) -> Path:
        """Manually generate a report.

        Args:
            circuit: The circuit object
            result: The simulation result
            format: Override format ("markdown", "html", "json")
            output_path: Custom output path

        Returns:
            Path to generated report
        """
        self._current_circuit = circuit
        data = self._build_report_data(result, None)

        fmt = (format or self._config["format"]).lower()

        if fmt == "html":
            return self._generate_html(data)
        elif fmt == "json":
            return self._generate_json(data)
        else:
            return self._generate_markdown(data)
