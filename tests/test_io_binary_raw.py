from __future__ import annotations

import struct
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from spicelab.io import load_dataset


def _write_binary_raw(
    path: Path,
    *,
    title: str,
    plotname: str,
    flags: str,
    vars_: Sequence[tuple[int, str, str]],
    points: Sequence[Sequence[float]],
) -> None:
    """Create a minimal binary RAW file compatible with our parser.

    points layout (per point):
      - flags == 'complex': first var real; each subsequent var has (re, im)
      - flags == 'real': each var single float
    """
    header_lines = [
        f"Title: {title}",
        "Date: Wed Sep 24 00:00:00 2025",
        f"Plotname: {plotname}",
        f"Flags: {flags}",
        f"No. Variables: {len(vars_)}",
        f"No. Points: {len(points)}",
        "Variables:",
    ]
    for idx, name, unit in vars_:
        header_lines.append(f"\t{idx}\t{name}\t{unit}")
    header_lines.append("Binary:")
    header = "\n".join(header_lines) + "\n"

    flat: list[float] = []
    complex_mode = "complex" in flags.lower()
    if complex_mode:
        for row in points:
            # row: (x, re1, im1, re2, im2, ...)
            flat.append(row[0])
            # remaining pairs
            for i in range(1, len(row), 2):
                flat.append(row[i])
                flat.append(row[i + 1])
    else:
        for row in points:
            flat.extend(row)

    blob = struct.pack(f"<{len(flat)}d", *flat)
    path.write_bytes(header.encode("utf-8") + blob)


def test_binary_raw_tran(tmp_path: Path) -> None:
    f = tmp_path / "tran_binary.raw"
    # Variables: time, v(out)
    vars_ = [(0, "time", "time"), (1, "v(out)", "voltage")]
    points: list[tuple[float, float]] = [
        (0.0, 0.0),
        (1e-6, 0.5),
        (2e-6, 1.0),
        (3e-6, 0.0),
    ]
    _write_binary_raw(
        f,
        title="Test Binary Tran",
        plotname="Transient Analysis",
        flags="real",
        vars_=vars_,
        points=points,
    )
    ds = load_dataset(f, allow_binary=True)
    assert ds.attrs.get("engine") == "ngspice"
    assert ds.attrs.get("analysis") == "tran"
    assert "time" in ds.coords
    assert "V(out)" in ds.data_vars
    np.testing.assert_allclose(ds["time"].values, [p[0] for p in points])
    np.testing.assert_allclose(ds["V(out)"].values, [p[1] for p in points])


def test_binary_raw_ac_complex(tmp_path: Path) -> None:
    f = tmp_path / "ac_binary.raw"
    # Variables: frequency, v(out) (complex)
    vars_ = [(0, "frequency", "frequency"), (1, "v(out)", "voltage")]
    # Each point: (freq, re, im)
    points: list[tuple[float, float, float]] = [
        (1.0, 1.0, 0.0),
        (10.0, 0.0, 1.0),
        (100.0, 0.70710678, 0.70710678),  # magnitude ~1
    ]
    _write_binary_raw(
        f,
        title="Test Binary AC",
        plotname="AC Analysis",
        flags="complex",
        vars_=vars_,
        points=[(p[0], p[1], p[2]) for p in points],
    )
    ds = load_dataset(f, allow_binary=True)
    assert ds.attrs.get("analysis") == "ac"
    assert "freq" in ds.coords
    mags_expected = [1.0, 1.0, 0.99999999]
    np.testing.assert_allclose(ds["V(out)"].values, mags_expected, rtol=1e-6, atol=1e-8)
    # Ensure coordinate frequencies correct
    np.testing.assert_allclose(ds["freq"].values, [p[0] for p in points])
