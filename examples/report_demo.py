"""Generate a sample HTML report demonstrating ReportBuilder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from spicelab.io.raw_reader import Trace, TraceSet
from spicelab.reporting import ReportBuilder
from spicelab.reporting.report import dataset_summary
from spicelab.viz.plotly import time_series_view

try:  # optional dependency; report still works without it
    import xarray as xr  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional
    xr = None  # type: ignore[assignment]


def _synthesise_response(
    rc: float, *, v_in: float, npoints: int = 400
) -> tuple[np.ndarray, np.ndarray]:
    time = np.linspace(0.0, 5.0 * rc, npoints)
    vout = v_in * (1.0 - np.exp(-time / rc))
    return time, vout


def main() -> None:
    resistance = 1_000.0  # ohms
    capacitance = 100e-9  # farads
    vin = 1.0
    tau = resistance * capacitance

    time, vout = _synthesise_response(tau, v_in=vin)
    traces = TraceSet(
        [
            Trace("time", "s", np.asarray(time)),
            Trace("V(out)", "V", np.asarray(vout)),
        ],
        meta={"description": "Synthetic RC charging curve"},
    )

    figure = time_series_view(
        traces,
        ys=["V(out)"],
        title="RC step response",
        xlabel="Time [s]",
        ylabel="Voltage [V]",
    )

    builder = ReportBuilder("RC Step Response", subtitle="Synthetic example report")
    summary = builder.add_section("Summary")
    summary.add_markdown(
        "This report showcases the :code:`ReportBuilder` helper with a synthetic "
        "RC charging curve rendered via Plotly."
    )
    summary.add_table(
        [
            {"parameter": "Resistance", "value": "1 kΩ"},
            {"parameter": "Capacitance", "value": "100 nF"},
            {"parameter": "τ", "value": f"{tau * 1e3:.1f} ms"},
        ],
        caption="Component values used to generate the trace",
    )

    if xr is not None:
        dataset = xr.Dataset({"V(out)": ("time", vout)}, coords={"time": time})
        summary.add_table(dataset_summary(dataset), caption="Dataset structure")

    waveform = builder.add_section("Waveform")
    waveform.add_figure(figure, caption="Output voltage versus time")

    output_dir = Path(__file__).resolve().parents[1] / "docs" / "assets" / "examples"
    outputs = builder.write(output_dir, slug="rc_report_demo", formats=("html",))
    html_path = outputs.get("html")
    if html_path:
        print(f"Report written to {html_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
