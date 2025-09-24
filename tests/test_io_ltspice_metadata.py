from __future__ import annotations

from pathlib import Path

from spicelab.io import load_dataset


def test_ltspice_binary_metadata(tmp_path: Path) -> None:
    # Copy provided fixture into tmp to avoid accidental mutation
    repo_fixture = Path("tests/fixtures/circuit_1.raw")
    target = tmp_path / "circuit_1.raw"
    target.write_bytes(repo_fixture.read_bytes())

    ds = load_dataset(target, allow_binary=True)

    # Engine heuristic should classify as ltspice (command/title contain LTspice)
    assert ds.attrs.get("engine") == "ltspice"
    assert ds.attrs.get("analysis") == "tran"
    # Offset captured
    assert "time_offset" in ds.attrs
    # Backannotation list preserved
    back = ds.attrs.get("backannotation")
    assert isinstance(back, list) and len(back) >= 1
    # Current signals aggregated
    currents = ds.attrs.get("current_signals")
    assert currents is None or isinstance(currents, list)
    # Subckt currents list
    subckt_currents = ds.attrs.get("subckt_currents")
    assert subckt_currents is None or isinstance(subckt_currents, list)
    # Basic coordinate
    assert "time" in ds.coords
    # Spot check a known voltage variable exists (V(vout) after normalization)
    assert any(name.startswith("V(") for name in ds.data_vars)
