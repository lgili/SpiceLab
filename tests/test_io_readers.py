from __future__ import annotations

from pathlib import Path

import pytest
from spicelab.io.readers import read_waveform, read_xyce_table


def _write_tmp(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_xyce_table_basic(tmp_path: Path) -> None:
    content = "time(s),V(out),I(R1)\n0,0,0\n1e-3,1,0.001\n2e-3,0.5,0.0005\n"
    p = _write_tmp(tmp_path, "wave.prn", content)
    ds = read_xyce_table(p)
    assert "time" in ds.coords
    assert set(ds.data_vars) == {"V(out)", "I(R1)"}
    assert float(ds["V(out)"].values[1]) == pytest.approx(1.0)


def test_waveform_dispatch_csv(tmp_path: Path) -> None:
    content = "freq, V(out)\n1,0.1\n10,0.2\n"
    p = _write_tmp(tmp_path, "bode.csv", content)
    ds = read_waveform(p)
    assert "freq" in ds.coords
    assert "V(out)" in ds
