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
    # Try to rename the independent variable to a conventional coord
    xname = ts.x.name.lower()
    if "time" in xname:
        ds = ds.rename({ts.x.name: "time"})
    elif "freq" in xname or "frequency" in xname:
        ds = ds.rename({ts.x.name: "freq"})
    return ds


def read(path: str | Path) -> Any:
    """Unified entry point. Dispatch by extension.

    - .raw → read_ltspice_raw
    - .prn/.csv (Xyce) and others will be added in next milestones.
    """

    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".raw":
        return read_ltspice_raw(p)
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


__all__ = ["read_ltspice_raw", "read", "read_ltspice"]
