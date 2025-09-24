from __future__ import annotations

import importlib
import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from .log_reader import parse_ngspice_log
from .ltspice_asc import circuit_from_asc
from .ltspice_parser import from_ltspice_file
from .raw_reader import parse_ngspice_ascii_raw_multi, parse_ngspice_raw


def _require_xarray() -> Any:
    try:
        xr = importlib.import_module("xarray")
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "xarray is required for unified readers. Install with: pip install xarray"
        ) from exc
    return xr


_TIME_ALIASES = {"t", "time", "tempo"}
_FREQ_ALIASES = {"f", "freq", "frequency"}

# Patterns for current / device quantity naming normalization
_CURRENT_PREFIXES = ("i(", "i(")  # already handled generically
_DEVICE_CURRENT_PATTERNS = [
    ("@", "[i]"),  # @R1[i]
    ("@", "[id]"),  # @M1[id]
]


def _normalize_device_current(name: str) -> str:
    """Normalize alternative device current notations.

    Patterns:
      - @R1[i]  -> I(R1)
      - @M1[id] -> Id(M1) (drain current explicit)
      - Generic fallback keeps original if pattern not recognized.
    """
    if name.startswith("@") and name.endswith("]") and "[" in name:
        body = name[1:-1]  # strip @ and trailing ]
        if "[" not in body:
            return name
        base, qual = body.split("[", 1)
        qual_low = qual.lower()
        if qual_low == "i":  # resistor/device generic current
            return f"I({base})"
        if qual_low == "id":  # MOS drain current
            return f"Id({base})"
    return name


def _normalize_signal_name(name: str) -> str:
    n = name.strip()
    low = n.lower()
    # Independent variable normalization handled elsewhere; here focus on signals
    if low.startswith("v(") or low.startswith("i("):
        return ("V" + n[1:]) if low.startswith("v(") else ("I" + n[1:])
    if low.startswith("v") and len(n) > 1 and n[1] == "(":  # v(out)
        return "V" + n[1:]
    if low.startswith("i") and len(n) > 1 and n[1] == "(":
        return "I" + n[1:]
    if low.startswith("ix("):
        # LTspice subcircuit current Ix(U1:1) → I(U1:1)
        return "I" + n[2:]
    # device current forms
    dev_norm = _normalize_device_current(n)
    return dev_norm


def _attach_common_attrs(
    ds: Any,
    *,
    engine: str,
    analysis: str | None,
    raw_path: str,
    **extra: Any,
) -> Any:
    ds.attrs.setdefault("engine", engine)
    if analysis:
        ds.attrs.setdefault("analysis", analysis)
    ds.attrs.setdefault("raw_path", str(raw_path))
    for k, v in extra.items():
        if v is not None:
            ds.attrs.setdefault(k, v)
    return ds


def _merge_steps(datasets: list[Any], coord_name: str = "step") -> Any:
    if len(datasets) == 1:
        return datasets[0]
    xr = _require_xarray()
    # Align variables
    base_vars = set(datasets[0].data_vars)
    for d in datasets[1:]:
        base_vars &= set(d.data_vars)
    if not base_vars:
        raise ValueError("No common variables across stepped datasets")
    # Reindex each to common vars
    aligned: list[Any] = []
    for i, d in enumerate(datasets):
        sub = d[list(base_vars)]
        sub = sub.expand_dims({coord_name: [i]})
        aligned.append(sub)
    return xr.concat(aligned, dim=coord_name)


def _infer_analysis_from_plot(plotname: str | None) -> str | None:
    if not plotname:
        return None
    pl = plotname.lower()
    for key in ("transient", "tran", "ac", "dc", "noise", "op"):
        if key in pl:
            if key == "transient":
                return "tran"
            return key
    return None


def _to_xarray_from_traceset(
    ts: Any,
    *,
    engine: str,
    analysis: str | None,
    raw_path: str,
    stepped: bool = False,
    step_index: int | None = None,
    total_steps: int | None = None,
    meta: dict[str, Any] | None = None,
    include_complex: tuple[str, ...] | None = None,
) -> Any:
    xr = _require_xarray()
    coords: dict[str, Any] = {"index": range(len(ts.x.values))}
    data_vars: dict[str, tuple[str, Any]] = {}
    current_signals: list[str] = []
    subckt_currents: list[str] = []
    for name in ts.names:
        trace_obj = ts[name]
        arr: Any = trace_obj.values if name != ts.x.name else trace_obj.values
        canon: str = _normalize_signal_name(name if name != ts.x.name else trace_obj.name)
        data_vars[canon] = ("index", arr)
        # Expand complex components if requested and available
        if include_complex and trace_obj._complex is not None:
            if "real" in include_complex:
                data_vars[f"{canon}_real"] = ("index", trace_obj.real())
            if "imag" in include_complex:
                data_vars[f"{canon}_imag"] = ("index", trace_obj.imag())
            if "phase" in include_complex or "phase_deg" in include_complex:
                data_vars[f"{canon}_phase_deg"] = ("index", trace_obj.phase_deg())
        if (
            canon.startswith("I(")
            and canon not in current_signals
            and canon not in {"I(time", "I(freq"}
        ):  # crude guard
            current_signals.append(canon)
        if name.lower().startswith("ix("):
            subckt_currents.append(canon)
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    # independent variable rename / coordinate extraction
    xname_low = ts.x.name.lower()
    if any(a in xname_low for a in _TIME_ALIASES):
        ds = ds.assign_coords({"time": ("index", ts.x.values)})
    elif any(a in xname_low for a in _FREQ_ALIASES):
        ds = ds.assign_coords({"freq": ("index", ts.x.values)})
    ds = _attach_common_attrs(ds, engine=engine, analysis=analysis, raw_path=str(raw_path))
    ds.attrs.setdefault("stepped", stepped)
    if total_steps is not None:
        ds.attrs.setdefault("steps", total_steps)
    if meta:
        for k, v in meta.items():
            if k not in ds.attrs:
                ds.attrs[k] = v
    if current_signals:
        ds.attrs.setdefault("current_signals", current_signals)
    if subckt_currents:
        ds.attrs.setdefault("subckt_currents", subckt_currents)
    if meta and "offset" in meta and "time" in ds.coords:
        ds.attrs.setdefault("time_offset", meta["offset"])
    # Derive analysis_kind alias
    if analysis and "analysis_kind" not in ds.attrs:
        ds.attrs["analysis_kind"] = analysis
    if step_index is not None:
        ds.attrs.setdefault("step_index", step_index)
    # Mark complex expansion info
    if include_complex:
        ds.attrs.setdefault("complex_components", list(include_complex))
    return ds


def read_ngspice_raw(path: str | Path, *, include_complex: tuple[str, ...] | None = None) -> Any:
    ts = parse_ngspice_raw(str(path))
    # parse single plot only; multi handled by read_ngspice_raw_multi
    # meta inference only available on ascii for now
    analysis = None
    try:
        # parse header manually for plotname
        with open(path, encoding="utf-8", errors="ignore") as f:
            for ln in f:
                if ln.lower().startswith("plotname:"):
                    analysis = _infer_analysis_from_plot(ln.split(":", 1)[1].strip())
                    break
    except Exception:
        pass
    # Fallback: if still unknown, attempt to use TraceSet meta (works for binary path)
    if analysis is None:
        meta = getattr(ts, "meta", None)
        if isinstance(meta, dict):  # defensive
            analysis = _infer_analysis_from_plot(meta.get("plotname"))
    return _to_xarray_from_traceset(
        ts,
        engine="ngspice",
        analysis=analysis,
        raw_path=str(path),
        meta=getattr(ts, "meta", None),
        include_complex=include_complex,
    )


def read_ngspice_raw_multi(
    path: str | Path, *, include_complex: tuple[str, ...] | None = None
) -> Any:
    tsets = parse_ngspice_ascii_raw_multi(str(path))
    if not tsets:
        raise ValueError("No plots found in RAW file")
    datasets: list[Any] = []
    for idx, ts in enumerate(tsets):
        ds = _to_xarray_from_traceset(
            ts,
            engine="ngspice",
            analysis=None,
            raw_path=str(path),
            stepped=True,
            step_index=idx,
            total_steps=len(tsets),
            meta=getattr(ts, "meta", None),
            include_complex=include_complex,
        )
        datasets.append(ds)
    if len(datasets) == 1:
        return datasets[0]
    merged = _merge_steps(datasets, coord_name="step")
    merged.attrs.update(datasets[0].attrs)
    return merged


def read_ltspice_raw(path: str | Path, *, include_complex: tuple[str, ...] | None = None) -> Any:
    # Reuse ngspice RAW parser (ASCII ou binário compatível simples)
    ds = read_ngspice_raw(path, include_complex=include_complex)
    # Adjust engine attr if heuristic suggests LTspice (presence of 'ltspice' in meta title)
    try:
        title = ds.attrs.get("title") or ds.attrs.get("plot_title")
        command = ds.attrs.get("command")
        if (title and "ltspice" in str(title).lower()) or (
            command and "ltspice" in str(command).lower()
        ):
            ds.attrs["engine"] = "ltspice"
    except Exception:  # pragma: no cover
        pass
    return ds


def read_ngspice_log(path: str | Path) -> Any:
    return parse_ngspice_log(path)


def read_xyce_table(path: str | Path) -> Any:
    """Read a simple table file (.csv or .prn) into an xarray.Dataset.

    Assumptions:
    - First non-empty, non-comment line contains column headers.
    - Subsequent lines contain numeric data; comments/blank lines are skipped.
    - Delimiter inferred: comma/semicolon -> CSV; otherwise whitespace.
    - First column is treated as the independent variable and set as a coordinate.
    - If the first column name matches time/freq, it's renamed to "time" or "freq".
    """

    xr = _require_xarray()

    p = Path(path)
    with open(p, encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f]

    # Filter out comments/empties
    content = [ln for ln in lines if ln and not ln.startswith(("#", ";", "*"))]
    if not content:
        raise ValueError("Empty table file")

    header = content[0]
    delim = "," if ("," in header or ";" in header) else None
    if delim is None:
        cols = header.split()
    else:
        hdr = header.replace(";", ",")
        cols = [c.strip() for c in hdr.split(",")]
    if len(cols) < 2:
        raise ValueError("Expected at least two columns in header")

    data_rows: list[list[float]] = []
    for row in content[1:]:
        if delim is None:
            parts = row.split()
        else:
            row_csv = row.replace(";", ",")
            parts = [c.strip() for c in row_csv.split(",")]
        if not parts:
            continue
        try:
            data_rows.append([float(tok) for tok in parts[: len(cols)]])
        except ValueError:
            # skip lines that aren't numeric (e.g., trailing comments)
            continue

    if not data_rows:
        raise ValueError("No numeric data rows found")

    # Parse potential units from header tokens like "time(s)" or "frequency,Hz"
    def split_name_unit(token: str) -> tuple[str, str | None]:
        tok = token.strip()
        low = tok.lower()
        # Do not treat signal names like v(out) / i(in) as (name, unit)
        if low.startswith("v(") or low.startswith("i("):
            return tok, None
        # time(s) or frequency(Hz)
        if ")" in tok and "(" in tok and tok.endswith(")"):
            name, unit = tok.rsplit("(", 1)
            unit_txt = unit[:-1].strip()  # drop ')'
            # Be conservative: if nested parentheses exist in name/unit, keep as name
            if "(" in name or ")" in name or "(" in unit_txt or ")" in unit_txt:
                return tok, None
            return name.strip(), unit_txt
        if "," in tok:
            name, unit = tok.split(",", 1)
            return name.strip(), unit.strip()
        return tok, None

    names_units = [split_name_unit(c) for c in cols]
    names = [nu[0] for nu in names_units]
    units = {nu[0]: nu[1] for nu in names_units if nu[1]}

    # Build arrays by column
    cols_data: dict[str, list[float]] = {name: [] for name in names}
    for r in data_rows:
        for i, name in enumerate(names):
            if i < len(r):
                cols_data[name].append(r[i])

    index = range(len(next(iter(cols_data.values()))))
    data_vars = {name: ("index", vals) for name, vals in cols_data.items()}
    ds = xr.Dataset(data_vars=data_vars, coords={"index": index})

    # Independent variable as coord + rename to conventional names when possible
    xname = names[0]
    try:
        ds = ds.set_coords(xname)
    except Exception:
        pass
    lx = xname.lower()
    if "time" in lx:
        ds = ds.rename({xname: "time"})
        if "time" in ds:
            ds = ds.set_coords("time")
    elif "freq" in lx or "frequency" in lx:
        ds = ds.rename({xname: "freq"})
        if "freq" in ds:
            ds = ds.set_coords("freq")
    # Attach units to attrs if present
    try:
        if units:
            ds.attrs.setdefault("units", {})
            for k, u in units.items():
                # ensure we map after any rename
                key = (
                    "time"
                    if (k == xname and "time" in ds)
                    else ("freq" if (k == xname and "freq" in ds) else k)
                )
                ds.attrs["units"][key] = u
    except Exception:
        pass
    return ds


def read_xyce_prn(path: str | Path) -> Any:
    return read_xyce_table(path)


def read_xyce_csv(path: str | Path) -> Any:
    return read_xyce_table(path)


def read(
    path: str | Path,
    *,
    engine_hint: str | None = None,
    allow_binary: bool = True,
    include_complex: tuple[str, ...] | None = None,
) -> Any:
    """Unified entry point for waveform-like outputs (.raw, .prn, .csv).

    Attempts to infer engine if not supplied. Multi-plot RAW (stepped) merged on step dim.
    """

    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".prn":
        return read_xyce_prn(p)
    if ext == ".csv":
        return read_xyce_csv(p)
    if ext == ".raw":
        # try multi first (ASCII only). If binary, fall back directly.
        try:
            if not _is_binary_raw(p):
                return read_ngspice_raw_multi(p, include_complex=include_complex)
        except Exception:
            pass
        if not allow_binary and _is_binary_raw(p):
            raise NotImplementedError(
                "Binary RAW detected but allow_binary=False. Pass allow_binary=True to load."
            )
        return read_ltspice_raw(p, include_complex=include_complex)
    raise NotImplementedError(f"Unsupported file type '{ext}'. Expected .raw/.prn/.csv")


def read_ltspice(path: str | Path) -> Any:
    """Unified LTspice loader for schematics/netlists → Circuit.

    - .asc → parse schematic and build a Circuit
    - .net/.cir → parse LTspice-exported netlist into a Circuit
    """

    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".asc":
        return circuit_from_asc(p)
    if ext in {".net", ".cir"}:
        return from_ltspice_file(p)
    raise NotImplementedError(f"Unsupported LTspice file '{p.name}'. Expected .asc/.net/.cir")


def read_waveform(
    path: str | Path,
    engine_hint: str | None = None,
    allow_binary: bool = True,
    include_complex: tuple[str, ...] | None = None,
) -> Any:  # alias
    return read(
        path, engine_hint=engine_hint, allow_binary=allow_binary, include_complex=include_complex
    )


def _is_binary_raw(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(256)
        return b"Binary:" in head or b"binary" in head
    except Exception:
        return False


def load_dataset(
    path: str | Path,
    *,
    engine: str | None = None,
    log: str | Path | None = None,
    netlist_hash: str | None = None,
    analysis_args: dict[str, object] | None = None,
    allow_binary: bool = False,
    complex_components: bool | tuple[str, ...] | list[str] | None = None,
) -> Any:
    """High-level façade returning an xarray.Dataset enriched with metadata.

    Parameters:
        path: waveform file (.raw/.prn/.csv)
        engine: optional explicit engine name (ngspice|ltspice|xyce)
        log: optional ngspice log path to extract version/warnings/errors
        netlist_hash: deterministic hash of circuit/netlist (for provenance)
        analysis_args: dict of analysis parameters used in run
        allow_binary: if False and RAW detected as binary, raise with guidance
        complex_components: True or iterable specifying which complex parts to expand
            (e.g. True → ("real","imag"), or ("real","imag","phase"))
    """

    p = Path(path)
    include_complex: tuple[str, ...] | None = None
    if complex_components:
        if complex_components is True:
            include_complex = ("real", "imag")
        elif isinstance(complex_components, list | tuple):
            include_complex = tuple(str(c).lower() for c in complex_components)
        else:  # unexpected truthy
            include_complex = ("real", "imag")

    ds = read_waveform(
        p,
        engine_hint=engine,
        allow_binary=allow_binary,
        include_complex=include_complex,
    )
    if engine:
        ds.attrs["engine"] = engine
    # Enrich with log summary if ngspice
    if log is not None and (
        engine == "ngspice" or (engine is None and ds.attrs.get("engine") == "ngspice")
    ):
        summary = read_ngspice_log(log)
        if summary.version:
            ds.attrs.setdefault("engine_version", summary.version)
        if summary.warnings:
            ds.attrs.setdefault("log_warnings", summary.warnings)
        if summary.errors:
            ds.attrs.setdefault("log_errors", summary.errors)
    if netlist_hash:
        ds.attrs.setdefault("netlist_hash", netlist_hash)
    if analysis_args:
        ds.attrs.setdefault("analysis_args", analysis_args)
    # Infer version for other engines if not set
    eng = ds.attrs.get("engine")
    if eng and "engine_version" not in ds.attrs:
        ver = _infer_engine_version(eng, p)
        if ver:
            ds.attrs["engine_version"] = ver
    # Additional derived metadata
    if "engine" in ds.attrs and "analysis" in ds.attrs:
        ds.attrs.setdefault("provenance", json.dumps({"loaded_at": datetime.utcnow().isoformat()}))
    # Count points
    try:
        ds.attrs.setdefault("n_points", int(ds.dims.get("index", 0)))
    except Exception:
        pass
    ds = normalize_dataset(ds)
    return ds


def to_polars(ds: Any) -> Any:
    """Return a `polars.DataFrame` with coordinates lifted to columns.

    Columns order heuristic:
      1. time or freq if present
      2. step (if present)
      3. remaining coords
      4. data variables (sorted)
    """
    try:
        pl = importlib.import_module("polars")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("polars is required for to_polars(); pip install polars") from exc
    cols: dict[str, Iterable[Any]] = {}
    # Collect coordinate-like columns
    coord_order: list[str] = []
    for special in ("time", "freq", "step"):
        if special in ds.coords:
            coord_order.append(special)
    # Include other 1D coords (excluding index)
    for c in ds.coords:
        if c not in coord_order and c != "index":
            coord_order.append(c)
    # Add coordinate values
    for c in coord_order:
        cols[c] = ds[c].values.tolist()
    # Data variables
    for name in sorted(ds.data_vars):
        cols[name] = ds[name].values.tolist()
    return pl.DataFrame(cols)


def to_pandas(ds: Any) -> Any:
    """Return a `pandas.DataFrame` with coords as columns (time/freq/step first)."""
    try:
        pd = importlib.import_module("pandas")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pandas is required for to_pandas(); pip install pandas") from exc
    data: dict[str, Any] = {}
    order: list[str] = []
    for special in ("time", "freq", "step"):
        if special in ds.coords:
            order.append(special)
    for c in ds.coords:
        if c not in order and c != "index":
            order.append(c)
    for c in order:
        data[c] = ds[c].values
    for name in sorted(ds.data_vars):
        data[name] = ds[name].values
    return pd.DataFrame(data)


def dataset_to_long_polars(ds: Any) -> Any:
    """Return a melted (long form) polars DataFrame: coord columns + name/value."""
    wide = to_polars(ds)
    coord_cols = [c for c in ("time", "freq", "step") if c in wide.columns]
    id_vars = coord_cols
    value_vars = [c for c in wide.columns if c not in id_vars]
    long = wide.unpivot(on=value_vars, index=id_vars, variable_name="signal", value_name="value")
    return long


def save_dataset(ds: Any, path: str | Path, *, format: str = "netcdf") -> Path:
    """Persist dataset to disk in selected format (netcdf|zarr|parquet)."""
    xr = _require_xarray()
    p = Path(path)
    fmt = format.lower()
    if fmt == "netcdf":
        xr.decode_cf(ds).to_netcdf(p)
    elif fmt == "zarr":  # pragma: no cover - optional
        ds.to_zarr(p, mode="w")
    elif fmt == "parquet":
        # Use long-form for parquet friendliness
        long_df = dataset_to_long_polars(ds)
        long_df.write_parquet(p)
    else:  # pragma: no cover
        raise ValueError("Unsupported format; choose netcdf|zarr|parquet")
    return p


def load_saved_dataset(path: str | Path) -> Any:
    p = Path(path)
    xr = _require_xarray()
    ext = p.suffix.lower()
    if ext in {".nc", ".netcdf"}:
        return xr.load_dataset(p)
    if ext == ".zarr":  # pragma: no cover
        return xr.open_zarr(p)
    if ext == ".parquet":
        pl = importlib.import_module("polars")
        df = pl.read_parquet(p)
        # reconstruct minimal dataset (wide pivot)
        if {"signal", "value"}.issubset(set(df.columns)):
            # pivot back
            coord_cols = [c for c in ("time", "freq", "step") if c in df.columns]
            wide = df.pivot(values="value", columns="signal", index=coord_cols)
            ds = _require_xarray().Dataset.from_dataframe(wide.to_pandas())
            return ds
        raise ValueError("Parquet file missing required columns 'signal' and 'value'")
    raise ValueError("Unsupported file extension for load_saved_dataset")


__all__ = [
    "read_waveform",
    "load_dataset",
    "normalize_dataset",
    "read_ltspice_raw",
    "read_ngspice_raw",
    "read_ngspice_raw_multi",
    "read_ngspice_log",
    "read_xyce_table",
    "read_xyce_prn",
    "read_xyce_csv",
    "read",
    "read_ltspice",
    "to_polars",
    "to_pandas",
    "dataset_to_long_polars",
    "save_dataset",
    "load_saved_dataset",
]


def list_signals(ds: Any) -> dict[str, list[str]]:
    """Return a classification of signals in an xarray Dataset.

    Categories:
      - voltage: names starting with 'V('
      - current: names starting with 'I(' (including device/subckt currents)
      - other: remaining data variables

    The returned dict maps category -> sorted list of variable names.

    Example
    -------
    >>> from spicelab.io import load_dataset, list_signals  # doctest: +SKIP
    >>> ds = load_dataset("example.raw", allow_binary=True)  # doctest: +SKIP
    >>> list_signals(ds)  # doctest: +SKIP
    {'voltage': ['V(out)'], 'current': ['I(R1)'], 'other': []}
    """
    volt: list[str] = []
    curr: list[str] = []
    other: list[str] = []
    for name in ds.data_vars:
        if name.startswith("V("):
            volt.append(name)
        elif name.startswith("I("):
            curr.append(name)
        else:
            other.append(name)
    return {
        "voltage": sorted(volt),
        "current": sorted(curr),
        "other": sorted(other),
    }


__all__.append("list_signals")


def _infer_engine_version(engine: str, path: Path) -> str | None:  # lightweight heuristics
    try:
        if engine == "ltspice":
            # Scan first 4KB for 'LTspice' pattern
            with open(path, "rb") as f:
                head = f.read(4096).decode("utf-8", errors="ignore")
            for line in head.splitlines():
                if "ltspice" in line.lower():
                    return line.strip()
            return None
        if engine == "xyce":
            # Xyce .prn/.csv às vezes têm header com versão (nossas fixtures não têm)
            with open(path, encoding="utf-8", errors="ignore") as f:
                first = f.readline().lower()
            if "xyce" in first and "version" in first:
                return first.strip()
            return None
    except Exception:
        return None
    return None


def normalize_dataset(ds: Any) -> Any:
    """Apply post-load normalization (idempotent).

    - Ensures signal names are canonical V(...)/I(...)
    - Re-derives time/freq coordinate if missing and an alias column exists
    """
    # If time coord absent, search for possible alias in data_vars
    if "time" not in ds.coords:
        for alias in _TIME_ALIASES:
            if alias in ds.data_vars:
                ds = ds.assign_coords({"time": ("index", ds[alias].values)})
                break
    if "freq" not in ds.coords:
        for alias in _FREQ_ALIASES:
            if alias in ds.data_vars:
                ds = ds.assign_coords({"freq": ("index", ds[alias].values)})
                break
    return ds
