from __future__ import annotations

import importlib
import os
import tempfile

import pytest
from spicelab.io.readers import read_ltspice_raw

ASCII_RAW_TEMPLATE = """Title:  transient
Date:   Thu Sep  1 12:00:00 2025
Plotname: Transient
Flags: real
No. Variables: 2
No. Points: {npoints}
Variables:
        0       time    time
        1       v(vout) voltage
Values:
{values}
"""


@pytest.mark.skipif(
    os.environ.get("SKIP_XARRAY_TESTS") == "1",
    reason="xarray not installed in env",
)
def test_read_ltspice_raw_smoke() -> None:
    try:
        xr = importlib.import_module("xarray")
    except Exception:
        pytest.skip("xarray not installed")

    import numpy as np

    t = np.linspace(0, 1e-6, 11)
    v = np.linspace(0, 1.0, 11)
    lines = [f"\t{i}\t{ti:.12g}\t{vi:.12g}" for i, (ti, vi) in enumerate(zip(t, v, strict=False))]

    with tempfile.TemporaryDirectory() as td:
        raw = os.path.join(td, "sim.raw")
        with open(raw, "w", encoding="utf-8") as f:
            f.write(ASCII_RAW_TEMPLATE.format(npoints=len(t), values="\n".join(lines)))
        ds = read_ltspice_raw(raw)
        assert isinstance(ds, xr.Dataset)
        # ensure variables are present
        assert any(k.lower().startswith("time") or k == "time" for k in ds.data_vars.keys())
        assert any(k.lower().startswith("v(") for k in ds.data_vars.keys())
