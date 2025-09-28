"""Orchestrator + measurements demo: sweep a parameter and export AC metrics.

Run from repo root:

  uv run --active python examples/orchestrator_ac_measurements.py
"""

from __future__ import annotations

from pathlib import Path

try:
    from spicelab.analysis import (
        GainBandwidthSpec,
        GainMarginSpec,
        PhaseMarginSpec,
        run_and_measure,
    )

    # Reuse CLI helpers for stable column ordering/sanitization
    from spicelab.cli.measure import _order_columns as _cli_order_columns  # type: ignore
    from spicelab.cli.measure import _sanitize_key as _cli_sanitize_key  # type: ignore
    from spicelab.core.circuit import Circuit
    from spicelab.core.components import VA, C, Resistor
    from spicelab.core.net import GND, Net
    from spicelab.core.types import AnalysisSpec, SweepSpec
    from spicelab.orchestrator import Job
except ModuleNotFoundError:  # local dev fallback
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    from spicelab.analysis import (
        GainBandwidthSpec,
        GainMarginSpec,
        PhaseMarginSpec,
        run_and_measure,
    )
    from spicelab.cli.measure import _order_columns as _cli_order_columns  # type: ignore
    from spicelab.cli.measure import _sanitize_key as _cli_sanitize_key  # type: ignore
    from spicelab.core.circuit import Circuit
    from spicelab.core.components import VA, C, Resistor
    from spicelab.core.net import GND, Net
    from spicelab.core.types import AnalysisSpec, SweepSpec
    from spicelab.orchestrator import Job


def build_rc() -> Circuit:
    c = Circuit("rc_ac")
    n_in = Net("vin")
    n_out = Net("out")
    vin = VA(ac_mag=1.0)
    r = Resistor("R1", "1k")
    cap = C("1u")
    c.add(vin, r, cap)
    c.connect(vin.ports[0], n_in)
    c.connect(vin.ports[1], GND)
    c.connect(r.ports[0], n_in)
    c.connect(r.ports[1], n_out)
    c.connect(cap.ports[0], n_out)
    c.connect(cap.ports[1], GND)
    return c


def main() -> int:
    circ = build_rc()
    ac = AnalysisSpec("ac", {"sweep_type": "dec", "n": 50, "fstart": 10.0, "fstop": 1e6})
    sweep = SweepSpec(variables={"R1": [1e3, 2e3, 5e3]})
    job = Job(circuit=circ, analyses=[ac], sweep=sweep, engine="ngspice")

    rows = run_and_measure(
        job,
        [
            PhaseMarginSpec(name="pm", numerator="V(out)", denominator="V(vin)"),
            GainBandwidthSpec(name="gbw", numerator="V(out)", denominator="V(vin)"),
            GainMarginSpec(name="gm", numerator="V(out)", denominator="V(vin)"),
        ],
        return_as="python",
    )

    out_dir = Path("examples_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "orchestrator_ac_metrics.csv"
    # Minimal CSV write without extra deps, with stable/sanitized headers
    if rows:
        # Build union of keys across rows
        seen: set[str] = set()
        all_keys: list[str] = []
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)
        # Order: param_* first (sorted), then ordered measurement fields
        param_cols = sorted([k for k in all_keys if isinstance(k, str) and k.startswith("param_")])
        measure_cols = [k for k in all_keys if k not in param_cols]
        ordered_measure = _cli_order_columns([c for c in measure_cols if isinstance(c, str)])
        cols = param_cols + ordered_measure
        header = [_cli_sanitize_key(c).replace("\r", "").replace("\n", "") for c in cols]
        with out_csv.open("w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for r in rows:
                f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"Exported {len(rows)} rows to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
