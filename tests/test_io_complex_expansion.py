from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from spicelab.io import load_dataset


def _write_complex_binary(path: Path) -> None:
    # Simple AC complex: frequency, V(out)
    header = (
        "Title: AC complex test\n"
        "Date: today\n"
        "Plotname: AC Analysis\n"
        "Flags: complex\n"
        "No. Variables: 2\n"
        "No. Points: 3\n"
        "Variables:\n"
        "\t0\tfrequency\tfrequency\n"
        "\t1\tv(out)\tvoltage\n"
        "Binary:\n"
    )
    # point-major layout: freq, (re,im)
    values = [
        10.0,
        1.0,
        0.0,  # mag 1, phase 0
        100.0,
        0.0,
        1.0,  # mag 1, phase 90
        1000.0,
        0.70710678,
        0.70710678,  # mag ~1, phase 45
    ]
    blob = struct.pack(f"<{len(values)}d", *values)
    path.write_bytes(header.encode() + blob)


def test_complex_expansion(tmp_path: Path) -> None:
    f = tmp_path / "ac_complex.raw"
    _write_complex_binary(f)
    ds = load_dataset(f, allow_binary=True, complex_components=("real", "imag", "phase"))
    assert "freq" in ds.coords
    assert "V(out)" in ds.data_vars
    # Expanded
    assert "V(out)_real" in ds and "V(out)_imag" in ds and "V(out)_phase_deg" in ds
    mags = ds["V(out)"].values
    re = ds["V(out)_real"].values
    im = ds["V(out)_imag"].values
    np.testing.assert_allclose(mags, np.sqrt(re**2 + im**2), rtol=1e-8)
