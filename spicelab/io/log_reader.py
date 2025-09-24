"""Helpers to parse ngspice textual logs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def read_errors(log_text: str) -> list[str]:
    return [ln for ln in log_text.splitlines() if "%error" in ln.lower()]


def read_warnings(log_text: str) -> list[str]:
    keys = ("warning:", "warn:")
    out: list[str] = []
    for ln in log_text.splitlines():
        s = ln.lower()
        if any(k in s for k in keys):
            out.append(ln)
    return out


def read_version(log_text: str) -> str | None:
    for ln in log_text.splitlines():
        s = ln.strip().lower()
        if s.startswith("ngspice") and ("version" in s or "ngspice" in s):
            return ln.strip()
    return None


@dataclass(frozen=True)
class NgSpiceLogSummary:
    version: str | None
    errors: list[str]
    warnings: list[str]


def parse_ngspice_log(path: str | Path) -> NgSpiceLogSummary:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")
    return NgSpiceLogSummary(
        version=read_version(text),
        errors=read_errors(text),
        warnings=read_warnings(text),
    )
