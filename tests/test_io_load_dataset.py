from __future__ import annotations

from pathlib import Path

import pytest
from spicelab.io.readers import load_dataset

# Synthetic ngspice-like RAW (single plot) with 'time' aliasing variations.
_RAW_TIME = """Title: single
Date: today
Plotname: Transient Analysis
Flags: real
No. Variables: 2
No. Points: 2
Variables:
\t0\tT\ttime
\t1\tv(out)\tvoltage
Values:
0\t0\t0
1\t1e-3\t1
"""

_LOG = """ngspice-40 version
Warning: w1
%ERROR: e1
"""


def _write(p: Path, name: str, text: str) -> Path:
    f = p / name
    f.write_text(text, encoding="utf-8")
    return f


def test_load_dataset_enrich(tmp_path: Path) -> None:
    raw = _write(tmp_path, "wave.raw", _RAW_TIME)
    log = _write(tmp_path, "run.log", _LOG)
    ds = load_dataset(
        raw,
        engine="ngspice",
        log=log,
        netlist_hash="abc123",
        analysis_args={"tstop": 1e-3},
    )
    assert ds.attrs.get("engine") == "ngspice"
    assert ds.attrs.get("engine_version") and "ngspice" in ds.attrs["engine_version"].lower()
    assert ds.attrs.get("netlist_hash") == "abc123"
    assert ds.attrs.get("analysis_args", {}).get("tstop") == 1e-3
    assert ds.attrs.get("log_warnings") and any("w1" in w for w in ds.attrs["log_warnings"])  # noqa: SIM115
    assert ds.attrs.get("log_errors") and any("e1" in e for e in ds.attrs["log_errors"])  # noqa: SIM115
    # Coordinate alias normalization: 'T' header should map to time
    assert "time" in ds.coords
    assert float(ds["V(out)"].values[-1]) == pytest.approx(1.0)


def test_load_dataset_binary_guard(tmp_path: Path) -> None:
    # Fake a binary header signature
    raw = _write(tmp_path, "bin.raw", "Binary: \n")
    with pytest.raises(NotImplementedError):
        load_dataset(raw)
