from __future__ import annotations

from pathlib import Path

from spicelab.io import list_signals, load_dataset


def test_list_signals_ltspice_fixture(tmp_path: Path) -> None:
    src = Path("tests/fixtures/circuit_1.raw")
    dst = tmp_path / "circuit_1.raw"
    dst.write_bytes(src.read_bytes())
    ds = load_dataset(dst, allow_binary=True)
    classified = list_signals(ds)
    assert set(classified.keys()) == {"voltage", "current", "other"}
    # Expect at least one voltage and several currents
    assert any(v.startswith("V(") for v in classified["voltage"])  # voltage signals present
    # Currents include device and subckt
    assert any(c.startswith("I(") for c in classified["current"])  # currents present
    # No unexpected duplication
    intersection = (
        set(classified["voltage"]) & set(classified["current"]) & set(classified["other"])
    )
    assert not intersection
