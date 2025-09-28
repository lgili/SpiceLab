"""Validate CLI CSV/schema ordering and header sanitization helpers.

This test doesn't execute ngspice or parse RAW files; it unit-tests the helpers
used by the CLI and orchestrator to ensure stable ordering and sanitized headers.
"""

from __future__ import annotations

from spicelab.cli.measure import _order_columns, _sanitize_key, _to_csv


def test_order_columns_core_known_then_rest():
    # Mix of expected core, known, and arbitrary keys in random order
    keys = [
        "random_b",
        "units",
        "denominator",
        "value",
        "freq",
        "measure",
        "type",
        "random_a",
        "tolerance_deg",
        "numerator",
    ]
    ordered = _order_columns(keys)
    # Core first
    assert ordered[:4] == [
        "measure",
        "type",
        "value",
        "units",
    ]
    # Then known in the specified order
    assert ordered[4:8] == [
        "freq",
        "numerator",
        "denominator",
        "tolerance_deg",
    ]
    # Remaining (sorted by sanitized name)
    assert ordered[8:] == sorted(["random_a", "random_b"], key=_sanitize_key)


def test_sanitize_key_removes_whitespace_and_newlines():
    assert _sanitize_key("toleran\nce\tdeg") == "toleran_ce_deg"
    assert _sanitize_key("  numerator  ") == "numerator"
    assert _sanitize_key("deno\rminator") == "deno_minator"


def test_to_csv_unions_keys_orders_and_sanitizes_headers():
    rows = [
        {
            "measure": "pm",
            "type": "ac",
            "value": 60.0,
            "units": "deg",
            "nume\nrator": "V(out)",  # intentional newline in key
            "denominator": "V(in)",
            "toleran\nce_deg": 1.0,  # intentional newline in key
            "freq": 1e3,
        },
        {
            "measure": "gbw",
            "type": "ac",
            "value": 1000.0,
            "units": "Hz",
        },
        {
            # Include param_* keys to ensure they appear first in orchestrator outputs
            "param_R1": 1000.0,
            "measure": "gm",
            "type": "ac",
            "value": -10.0,
            "units": "dB",
        },
    ]

    csv_text = _to_csv(rows)
    header = csv_text.splitlines()[0]
    # Confirm header has no raw CR/LF artifacts within column names
    assert "\n" not in header and "\r" not in header
    # Confirm known columns present and sanitized (whitespace collapsed to underscores)
    assert "toleran_ce_deg" in header
    assert "nume_rator" in header  # sanitization collapsed the newline with underscore
    # Core columns come first
    expected_core_prefix = "measure,type,value,units"
    assert header.startswith(expected_core_prefix)
