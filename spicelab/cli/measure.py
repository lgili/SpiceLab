from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any

from ..analysis.measure import (
    ENOBSpec,
    GainBandwidthSpec,
    GainMarginSpec,
    PhaseMarginSpec,
    RiseTimeSpec,
    THDSpec,
    measure,
)
from ..io.readers import load_dataset


def _sanitize_key(k: str) -> str:
    # Replace any whitespace (including newlines) with underscore
    return "_".join(str(k).split())


def _order_columns(keys: list[str]) -> list[str]:
    core = ["measure", "type", "value", "units"]
    known = [
        "freq",
        "numerator",
        "denominator",
        "tolerance_deg",
        "signal",
        "harmonics",
        "f0",
        "low_pct",
        "high_pct",
        "sinad_db",
    ]
    # Match by sanitized name
    by_sanitized = {_sanitize_key(k): k for k in keys}
    rest = [k for k in keys if _sanitize_key(k) not in set(core + known)]
    ordered: list[str] = []
    # Core first
    for name in core:
        if name in by_sanitized:
            ordered.append(by_sanitized[name])
    # Then known in that order
    for name in known:
        if name in by_sanitized and by_sanitized[name] not in ordered:
            ordered.append(by_sanitized[name])
    # Then the remaining (sorted by sanitized for stability)
    remaining = sorted(rest, key=lambda k: _sanitize_key(k))
    ordered.extend([k for k in remaining if k not in ordered])
    return ordered


def _to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    # Build stable, unioned columns across all rows
    seen: set[str] = set()
    all_keys: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                all_keys.append(k)
    ordered_keys = _order_columns(all_keys)
    header_names = [_sanitize_key(k).replace("\r", "").replace("\n", "") for k in ordered_keys]
    out: list[str] = [",".join(header_names)]
    for r in rows:
        out.append(",".join(str(r.get(k, "")) for k in ordered_keys))
    return "\n".join(out)


def _available_signals(ds: Any) -> list[str]:
    try:
        return list(getattr(ds, "data_vars", {}).keys())
    except Exception:
        # Fallback: attempt attributes used by TraceSet-like
        names = getattr(getattr(ds, "traces", None), "names", None)
        return list(names) if names else []


def _axis_name_of(var: Any) -> str:
    preferred = ("time", "freq", "frequency")
    coords = getattr(var, "coords", {}) or {}
    for key in preferred:
        if key in coords:
            return key
    dims = getattr(var, "dims", ()) or ()
    if dims:
        return str(dims[0])
    return "index"


def _signals_details(ds: Any) -> list[dict[str, str]]:
    details: list[dict[str, str]] = []
    try:
        dv = getattr(ds, "data_vars", {})
        for name, var in dv.items():
            values = getattr(var, "values", None)
            axis = _axis_name_of(var)
            try:
                import numpy as _np  # local import to keep CLI fast

                arr = values if values is not None else ()
                dtype = "complex" if _np.iscomplexobj(arr) else "real"
            except Exception:
                dtype = "unknown"
            ax_norm = (
                "freq"
                if axis.lower() in ("freq", "frequency")
                else ("time" if axis.lower() == "time" else axis)
            )
            details.append({"name": str(name), "dtype": dtype, "axis": ax_norm})
    except Exception:
        pass
    return details


def _resolve_name(name: str, available: list[str]) -> str | None:
    m = {n.lower(): n for n in available}
    return m.get(name.lower())


def _suggest(name: str, available: list[str], limit: int = 8) -> list[str]:
    # Combine fuzzy match and substring contains
    low = name.lower()
    subs = [n for n in available if low in n.lower()]
    fuzzy = difflib.get_close_matches(name, available, n=limit, cutoff=0.5)
    out: list[str] = []
    for n in subs + fuzzy:
        if n not in out:
            out.append(n)
        if len(out) >= limit:
            break
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="spicelab-measure",
        description=(
            "Run AC/TRAN measurements on a supported dataset file (NGSpice RAW, "
            "LTspice RAW, Xyce PRN) and export results."
        ),
    )
    p.add_argument(
        "raw", type=Path, help="Path to a dataset file (NGSpice .raw, LTspice .raw, Xyce .prn)"
    )
    p.add_argument(
        "--list-signals", action="store_true", help="List available signals in RAW and exit"
    )
    p.add_argument(
        "--list-details",
        action="store_true",
        help="When used with --list-signals, include dtype (real/complex) and axis (time/freq)",
    )
    p.add_argument("--ac", action="store_true", help="Enable AC metrics (PM, GBW, GM)")
    p.add_argument("--tran", action="store_true", help="Enable TRAN metrics (rise, THD, ENOB)")
    p.add_argument("--num", default="V(out)", help="AC numerator (default: V(out))")
    p.add_argument("--den", default="V(in)", help="AC denominator (default: V(in))")
    p.add_argument("--signal", default="V(out)", help="TRAN signal (default: V(out))")
    p.add_argument("--f0", type=float, default=None, help="Fundamental frequency for THD/ENOB")
    p.add_argument("--harmonics", type=int, default=5, help="Number of harmonics for THD/ENOB")
    p.add_argument(
        "--format",
        choices=["json", "csv", "schema"],
        default="json",
        help="Output format (schema prints only the CSV header/columns)",
    )
    p.add_argument("--out", type=Path, default=None, help="Output file path (default: stdout)")
    p.add_argument(
        "--warn-nonfinite", action="store_true", help="Warn if any metric value is NaN/Inf"
    )
    p.add_argument(
        "--fail-on-nonfinite",
        action="store_true",
        help="Exit with error if any metric value is NaN/Inf",
    )
    args = p.parse_args(argv)

    ds = load_dataset(args.raw, allow_binary=True)

    # Just list signals and exit
    if args.list_signals:
        if args.list_details:
            det = _signals_details(ds)
            if args.format == "json":
                data = json.dumps({"signals": det}, indent=2)
            else:
                lines: list[str] = ["name,dtype,axis"]
                for d in det:
                    lines.append(f"{d['name']},{d['dtype']},{d['axis']}")
                data = "\n".join(lines)
        else:
            names = _available_signals(ds)
            if args.format == "json":
                data = json.dumps({"signals": names}, indent=2)
            else:
                data = "\n".join(names)
        if args.out:
            args.out.write_text(data + ("\n" if not data.endswith("\n") else ""), encoding="utf-8")
        else:
            sys.stdout.write(data + ("\n" if not data.endswith("\n") else ""))
        return 0

    if not args.ac and not args.tran:
        # default to both
        args.ac = True
        args.tran = True

    # Validate provided signal names and map to actual case
    available = _available_signals(ds)
    # AC: numerator/denominator
    if args.ac:
        resolved_num = _resolve_name(args.num, available)
        resolved_den = _resolve_name(args.den, available)
        if resolved_num is None:
            sugg = _suggest(args.num, available)
            msg = (
                f"error: AC numerator '{args.num}' not found. Try: "
                f"{', '.join(sugg) or 'no suggestions'}\n"
            )
            sys.stderr.write(msg)
            return 2
        if resolved_den is None:
            sugg = _suggest(args.den, available)
            msg = (
                f"error: AC denominator '{args.den}' not found. Try: "
                f"{', '.join(sugg) or 'no suggestions'}\n"
            )
            sys.stderr.write(msg)
            return 2
        args.num, args.den = resolved_num, resolved_den
    # TRAN: signal
    if args.tran:
        resolved_sig = _resolve_name(args.signal, available)
        if resolved_sig is None:
            sugg = _suggest(args.signal, available)
            msg = (
                f"error: TRAN signal '{args.signal}' not found. Try: "
                f"{', '.join(sugg) or 'no suggestions'}\n"
            )
            sys.stderr.write(msg)
            return 2
        args.signal = resolved_sig

    specs: list[Any] = []
    if args.ac:
        specs += [
            PhaseMarginSpec(name="pm", numerator=args.num, denominator=args.den),
            GainBandwidthSpec(name="gbw", numerator=args.num, denominator=args.den),
            GainMarginSpec(name="gm", numerator=args.num, denominator=args.den),
        ]
    if args.tran:
        specs += [
            RiseTimeSpec(name="tr", signal=args.signal),
            THDSpec(name="thd", signal=args.signal, harmonics=args.harmonics, f0=args.f0),
            ENOBSpec(name="enob", signal=args.signal, harmonics=args.harmonics, f0=args.f0),
        ]

    # Fast path: if only the schema is requested, synthesize header without executing measurements
    if args.format == "schema":
        keys: set[str] = set(["measure", "type", "value", "units"])  # core
        if args.ac:
            keys.update(["freq", "numerator", "denominator", "tolerance_deg"])
        if args.tran:
            # Include fields relevant to RiseTime, THD, ENOB
            keys.update(["signal", "harmonics", "f0", "low_pct", "high_pct", "sinad_db"])
        ordered_keys2 = _order_columns(sorted(keys))
        header2 = [_sanitize_key(k).replace("\r", "").replace("\n", "") for k in ordered_keys2]
        data = ",".join(header2)
        if args.out:
            args.out.write_text(data, encoding="utf-8")
        else:
            sys.stdout.write(data + ("\n" if not data.endswith("\n") else ""))
        return 0

    # Run specs, converting known runtime ValueErrors into NaN rows rather than aborting
    rows: list[dict[str, Any]] = []
    for sp in specs:
        try:
            part = measure(ds, [sp], return_as="python")
            rows.extend(part)
        except KeyError as e:
            msg = str(e)
            sys.stderr.write(f"error: {msg}\n")
            return 2
        except ValueError:
            # Gracefully degrade to NaN for this spec

            if isinstance(sp, ENOBSpec):
                rows.append(
                    {
                        "measure": sp.name,
                        "type": "enob",
                        "value": float("nan"),
                        "units": "bits",
                        "signal": sp.signal,
                        "sinad_db": float("nan"),
                        "f0": float(sp.f0) if sp.f0 is not None else float("nan"),
                    }
                )
            elif isinstance(sp, THDSpec):
                rows.append(
                    {
                        "measure": sp.name,
                        "type": "thd",
                        "value": float("nan"),
                        "units": "%",
                        "signal": sp.signal,
                        "harmonics": float(sp.harmonics),
                        "f0": float(sp.f0) if sp.f0 is not None else float("nan"),
                    }
                )
            elif isinstance(sp, RiseTimeSpec):
                rows.append(
                    {
                        "measure": sp.name,
                        "type": "rise_time",
                        "value": float("nan"),
                        "units": "s",
                        "signal": sp.signal,
                        "low_pct": float(sp.low_pct),
                        "high_pct": float(sp.high_pct),
                    }
                )
            elif isinstance(sp, PhaseMarginSpec):
                rows.append(
                    {
                        "measure": sp.name,
                        "type": "phase_margin",
                        "value": float("nan"),
                        "units": "deg",
                        "freq": float("nan"),
                        "numerator": sp.numerator,
                        "denominator": sp.denominator,
                        "tolerance_deg": float("nan"),
                    }
                )
            elif isinstance(sp, GainBandwidthSpec):
                rows.append(
                    {
                        "measure": sp.name,
                        "type": "gbw",
                        "value": float("nan"),
                        "units": "Hz",
                        "numerator": sp.numerator,
                        "denominator": sp.denominator,
                    }
                )
            elif isinstance(sp, GainMarginSpec):
                rows.append(
                    {
                        "measure": sp.name,
                        "type": "gain_margin",
                        "value": float("nan"),
                        "units": "dB",
                        "freq": float("nan"),
                        "numerator": sp.numerator,
                        "denominator": sp.denominator,
                        "tolerance_deg": float("nan"),
                    }
                )
            else:  # unknown spec type: generic NaN
                rows.append(
                    {
                        "measure": getattr(sp, "name", "unknown"),
                        "type": "unknown",
                        "value": float("nan"),
                        "units": "",
                    }
                )

    # Non-finite handling
    if any(
        not (
            isinstance(r.get("value"), int | float)
            and (r["value"] == r["value"] and r["value"] not in (float("inf"), float("-inf")))
        )
        for r in rows
    ):
        # More robust: use math to test finiteness
        import math as _math

        nonf = [
            r
            for r in rows
            if not (
                isinstance(r.get("value"), int | float)
                and _math.isfinite(r.get("value", float("nan")))
            )
        ]
        if nonf and args.warn_nonfinite:
            for r in nonf:
                msg = (
                    "warning: metric '"
                    + str(r.get("measure"))
                    + "' returned non-finite value: "
                    + str(r.get("value"))
                    + "\n"
                )
                sys.stderr.write(msg)
        if nonf and args.fail_on_nonfinite:
            return 3

    if args.format == "json":
        data = json.dumps(rows, indent=2)
    else:
        data = _to_csv(rows)

    if args.out:
        args.out.write_text(data, encoding="utf-8")
    else:
        sys.stdout.write(data + ("\n" if not data.endswith("\n") else ""))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
