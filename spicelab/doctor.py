"""Environment diagnostic helpers for spicelab."""

from __future__ import annotations

import os
import platform
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = ["CheckResult", "collect_diagnostics", "format_cli", "main", "shutil"]


@dataclass(frozen=True)
class CheckResult:
    target: str
    status: str  # ``ok`` | ``warn`` | ``missing``
    detail: str | None = None
    hint: str | None = None


def collect_diagnostics() -> list[CheckResult]:
    results: list[CheckResult] = []
    results.extend(_check_engine("ngspice", ["SPICELAB_NGSPICE"], default_names=["ngspice"]))
    results.extend(
        _check_engine(
            "ltspice",
            ["SPICELAB_LTSPICE", "LTSPICE_EXE"],
            default_names=["LTspice", "XVIIx64.exe", "XVIIx86.exe"],
        )
    )
    results.extend(_check_engine("xyce", ["SPICELAB_XYCE"], default_names=["Xyce"]))
    results.append(_check_ngspice_shared())
    return results


def format_cli(results: Sequence[CheckResult]) -> str:
    lines = ["spicelab environment check"]
    for item in results:
        icon = {"ok": "✔", "warn": "⚠", "missing": "✖"}.get(item.status, "•")
        detail = f" – {item.detail}" if item.detail else ""
        lines.append(f" {icon} {item.target}{detail}")
        if item.hint:
            lines.append(f"    hint: {item.hint}")
    return "\n".join(lines)


def main() -> int:  # pragma: no cover - thin CLI wrapper
    results = collect_diagnostics()
    print(format_cli(results))
    return 0 if all(r.status == "ok" for r in results if r.status != "warn") else 1


# ---------------------------------------------------------------------------
def _check_engine(
    name: str, env_vars: Iterable[str], *, default_names: Iterable[str]
) -> list[CheckResult]:
    matches: list[CheckResult] = []
    for env in env_vars:
        value = os.environ.get(env)
        if value:
            path = Path(value)
            if path.exists():
                matches.append(CheckResult(target=f"{name} ({env})", status="ok", detail=str(path)))
            else:
                matches.append(
                    CheckResult(
                        target=f"{name} ({env})",
                        status="missing",
                        detail=str(path),
                        hint="Path set but file not found",
                    )
                )
    for candidate in default_names:
        exe = shutil.which(candidate)
        if exe:
            matches.append(CheckResult(target=f"{name}", status="ok", detail=exe))
            break
    if not matches:
        hint = _engine_hint(name)
        matches.append(CheckResult(target=name, status="missing", hint=hint))
    return matches


def _check_ngspice_shared() -> CheckResult:
    env = os.environ.get("SPICELAB_NGSPICE_SHARED")
    if env:
        path = Path(env)
        if path.exists():
            return CheckResult("libngspice", status="ok", detail=str(path))
        return CheckResult(
            "libngspice",
            status="missing",
            detail=str(path),
            hint="Environment variable set but library not found",
        )

    # Probe common locations without failing hard
    from spicelab.spice import ngspice_shared_backend as backend

    for candidate in getattr(backend, "_candidate_paths", lambda: [])():
        if candidate and Path(candidate).exists():
            return CheckResult("libngspice", status="ok", detail=str(candidate))

    hint = "Install libngspice (e.g. brew install libngspice or apt install libngspice0-dev)"
    return CheckResult("libngspice", status="warn", hint=hint)


def _engine_hint(name: str) -> str:
    system = platform.system().lower()
    if name == "ngspice":
        if "darwin" in system:
            return "brew install ngspice"
        if "windows" in system:
            return "Install from ngspice.sourceforge.io"
        return "sudo apt install ngspice"
    if name == "ltspice":
        if "darwin" in system:
            return "brew install --cask ltspice"
        return "Download LTspice from Analog Devices"
    if name == "xyce":
        return "Get binaries from https://xyce.sandia.gov"
    return "Install the missing engine and set the SPICELAB_* variable if needed"


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
