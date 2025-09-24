from __future__ import annotations

import importlib
import os
import tempfile

import pytest
from spicelab.io.readers import read_ltspice_raw, read_xyce_table

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

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".raw", delete=False, mode="w", encoding="utf-8"
        ) as f:
            f.write(ASCII_RAW_TEMPLATE.format(npoints=len(t), values="\n".join(lines)))
            tmp_path = f.name
        assert os.path.exists(tmp_path)
        ds = read_ltspice_raw(tmp_path)
        assert isinstance(ds, xr.Dataset)
        # ensure variables are present
        keys = set(ds.data_vars.keys()) | set(ds.coords.keys())
        assert any(k.lower().startswith("time") or k == "time" for k in keys)
        assert any(k.lower().startswith("v(") for k in ds.data_vars.keys())
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@pytest.mark.skipif(
    os.environ.get("SKIP_XARRAY_TESTS") == "1",
    reason="xarray not installed in env",
)
def test_read_xyce_table_csv_and_prn() -> None:
    try:
        xr = importlib.import_module("xarray")
    except Exception:
        pytest.skip("xarray not installed")

    csv_text = "time,v(out)\n0,0\n1e-6,0.1\n2e-6,0.2\n"
    prn_text = "frequency v(out)\n1e1 0.0\n1e2 0.5\n1e3 1.0\n"

    tmp_csv = None
    tmp_prn = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w", encoding="utf-8"
        ) as f1:
            f1.write(csv_text)
            tmp_csv = f1.name
        with tempfile.NamedTemporaryFile(
            suffix=".prn", delete=False, mode="w", encoding="utf-8"
        ) as f2:
            f2.write(prn_text)
            tmp_prn = f2.name

        ds_csv = read_xyce_table(tmp_csv)
        ds_prn = read_xyce_table(tmp_prn)
        assert isinstance(ds_csv, xr.Dataset)
        assert isinstance(ds_prn, xr.Dataset)
        # Ensure coords present
        keys_csv = set(ds_csv.data_vars.keys()) | set(ds_csv.coords.keys())
        keys_prn = set(ds_prn.data_vars.keys()) | set(ds_prn.coords.keys())
        assert any(k.lower().startswith("time") or k == "time" for k in keys_csv)
        assert any(
            k.lower().startswith("freq") or k == "freq" or k.lower().startswith("frequency")
            for k in keys_prn
        )
        assert any(k.lower().startswith("v(") for k in ds_csv.data_vars.keys())
        assert any(k.lower().startswith("v(") for k in ds_prn.data_vars.keys())
    finally:
        for p in (tmp_csv, tmp_prn):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass
