from __future__ import annotations

from pathlib import Path

from spicelab.io import load_dataset

_RAW = """Title: device currents
Date: today
Plotname: Transient Analysis
Flags: real
No. Variables: 4
No. Points: 2
Variables:
\t0\ttime\ttime
\t1\t@R1[i]\tcurrent
\t2\t@M2[id]\tcurrent
\t3\tv(out)\tvoltage
Values:
0\t0\t0.001\t0.01\t0.0
1\t1e-3\t0.002\t0.02\t1.0
"""


def test_device_current_normalization(tmp_path: Path) -> None:
    raw = tmp_path / "dev_curr.raw"
    raw.write_text(_RAW)
    ds = load_dataset(raw)
    # Expect I(R1) and Id(M2)
    assert "I(R1)" in ds.data_vars
    assert "Id(M2)" in ds.data_vars
    # Basic shape
    assert len(ds["I(R1)"].values) == 2
    assert float(ds["Id(M2)"].values[-1]) == 0.02
