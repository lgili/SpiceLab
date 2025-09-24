from __future__ import annotations

from pathlib import Path

import pytest
from spicelab.io.readers import read_ngspice_log, read_ngspice_raw_multi

# Minimal synthetic multi-plot RAW (two blocks). This is a simplified structure derived
# from typical ngspice ASCII output; sufficient for parser exercise.
_MULTI_RAW = """Title: synthetic stepped
Date: today
Plotname: Transient Analysis
Flags: real
No. Variables: 2
No. Points: 2
Variables:
	0	time	time
	1	v(out)	voltage
Values:
0	0	0
1	1e-3	1.0
Title: synthetic stepped
Date: today
Plotname: Transient Analysis
Flags: real
No. Variables: 2
No. Points: 2
Variables:
	0	time	time
	1	v(out)	voltage
Values:
0	0	0.5
1	1e-3	0.8
"""

_LOG_TXT = """ngspice-42 version
Warning: dummy warning line
%ERROR: dummy error line
"""


def _write(p: Path, name: str, text: str) -> Path:
    f = p / name
    f.write_text(text, encoding="utf-8")
    return f


def test_multiplot_merge(tmp_path: Path) -> None:
    raw = _write(tmp_path, "multi.raw", _MULTI_RAW)
    ds = read_ngspice_raw_multi(raw)
    assert ds.attrs.get("stepped") is True
    assert ds.dims.get("step") == 2
    # Expect time coordinate present
    assert "time" in ds.coords
    # Two steps: value difference
    v = ds["V(out)"].isel(index=1)
    assert float(v.sel(step=0).values) == pytest.approx(1.0)
    assert float(v.sel(step=1).values) == pytest.approx(0.8)


def test_log_summary(tmp_path: Path) -> None:
    log = _write(tmp_path, "run.log", _LOG_TXT)
    summary = read_ngspice_log(log)
    assert summary.version and "ngspice" in summary.version.lower()
    assert any("warning" in w.lower() for w in summary.warnings)
    assert any("error" in e.lower() for e in summary.errors)
