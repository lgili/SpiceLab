from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .ltspice_asc import circuit_from_asc
from .ltspice_parser import from_ltspice_file
from .raw_reader import parse_ngspice_raw


def _require_xarray() -> Any:
    try:
        xr = importlib.import_module("xarray")
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "xarray is required for unified readers. Install with: pip install xarray"
        ) from exc
    return xr


def read_ltspice_raw(path: str | Path) -> Any:
    """Read an LTspice/NGSpice ASCII RAW file into an xarray.Dataset.

    Currently relies on the existing TraceSet parser and converts to xarray.
    Binary RAW is not supported yet (use ASCII mode in the generator).
    """

    xr = _require_xarray()
    ts = parse_ngspice_raw(str(path))
    coords = {"index": range(len(ts.x.values))}
    data_vars = {
        name: ("index", (ts[name].values if name != ts.x.name else ts.x.values))
        for name in ts.names
    }
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    # Mark the independent variable as a coordinate, then try conventional renames
    try:
        ds = ds.set_coords(ts.x.name)
    except Exception:
        pass
    # Try to rename the independent variable to a conventional coord
    xname = ts.x.name.lower()
    if "time" in xname:
        ds = ds.rename({ts.x.name: "time"})
        if "time" in ds:
            ds = ds.set_coords("time")
    elif "freq" in xname or "frequency" in xname:
        ds = ds.rename({ts.x.name: "freq"})
        if "freq" in ds:
            ds = ds.set_coords("freq")
    return ds


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


def read(path: str | Path) -> Any:
    """Unified entry point. Dispatch by extension.

    - .raw → read_ltspice_raw
    - .prn/.csv (Xyce-like tables) → read_xyce_table
    """

    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".raw":
        return read_ltspice_raw(p)
    if ext in {".csv", ".prn"}:
        return read_xyce_table(p)
    raise NotImplementedError(f"Unsupported file type '{ext}'. Expected .raw for now.")


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


__all__ = ["read_ltspice_raw", "read", "read_ltspice", "read_xyce_table"]
