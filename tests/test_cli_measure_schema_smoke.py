from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_cli(args: list[str]) -> str:
    cmd = [sys.executable, "-m", "spicelab.cli.measure", *args]
    out = subprocess.check_output(cmd, cwd=Path.cwd())
    return out.decode("utf-8")


def test_cli_list_details_csv_header_and_rows():
    raw = Path("tests/fixtures/rc_ac_ng.raw")
    text = _run_cli([str(raw), "--list-signals", "--list-details", "--format", "csv"]).strip()
    lines = text.splitlines()
    assert lines, "expected output lines"
    # Header exactly
    assert lines[0] == "name,dtype,axis"
    # No CR/LF artifacts within columns
    assert "\r" not in lines[0]
    # At least one row
    assert len(lines) > 1


def test_cli_schema_ac_ordering_and_no_crlf():
    raw = Path("tests/fixtures/rc_ac_ng.raw")
    # Fixture exposes V(out) and I(R1); use those for numerator/denominator
    text = _run_cli(
        [str(raw), "--ac", "--num", "V(out)", "--den", "I(R1)", "--format", "schema"]
    ).strip()
    # Core prefix
    assert text.startswith("measure,type,value,units")
    # Known AC fields appear
    assert "freq" in text
    assert "numerator" in text
    assert "denominator" in text
    # tolerance_deg may or may not appear depending on spec defaults; don't require it
    # No embedded CR
    assert "\r" not in text


def test_cli_schema_tran_includes_signal_fields():
    raw = Path("tests/fixtures/rc_tran_ng.raw")
    text = _run_cli([str(raw), "--tran", "--signal", "V(out)", "--format", "schema"]).strip()
    # Core prefix
    assert text.startswith("measure,type,value,units")
    # Known TRAN fields
    assert "signal" in text
    assert "harmonics" in text
    assert "f0" in text
    # No embedded CR
    assert "\r" not in text


def test_cli_schema_ac_other_variants():
    # LTspice fixture
    text_lt = _run_cli(
        [
            "tests/fixtures/rc_ac_lt.raw",
            "--ac",
            "--num",
            "V(out)",
            "--den",
            "I(R1)",
            "--format",
            "schema",
        ]
    ).strip()
    assert text_lt.startswith("measure,type,value,units")
    assert "freq" in text_lt and "numerator" in text_lt and "denominator" in text_lt
    assert "\r" not in text_lt
    # Xyce fixture
    text_xy = _run_cli(
        [
            "tests/fixtures/rc_ac_xy.prn",
            "--ac",
            "--num",
            "V(out)",
            "--den",
            "I(R1)",
            "--format",
            "schema",
        ]
    ).strip()
    assert text_xy.startswith("measure,type,value,units")
    assert "freq" in text_xy and "numerator" in text_xy and "denominator" in text_xy
    assert "\r" not in text_xy


def test_cli_list_details_other_variants():
    # Ensure header and at least one row for each variant
    for path in [
        "tests/fixtures/rc_ac_ng.raw",
        "tests/fixtures/rc_ac_lt.raw",
        "tests/fixtures/rc_ac_xy.prn",
    ]:
        txt = _run_cli([path, "--list-signals", "--list-details", "--format", "csv"]).strip()
        lines = txt.splitlines()
        assert lines and lines[0] == "name,dtype,axis"
        assert len(lines) > 1
        assert "\r" not in txt
