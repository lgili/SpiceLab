"""Environment diagnostic and interactive helper for spicelab.

The "doctor" can:
- Run local environment checks (engines, shared libraries)
- Optionally, answer questions or propose remediation via an LLM if configured
"""

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


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper
    import argparse

    p = argparse.ArgumentParser(prog="spicelab-doctor", description="Environment checks and help")
    p.add_argument("--ask", help="Ask the doctor for help (LLM-backed)")
    p.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format for checks"
    )
    p.add_argument("--model", help="Override OPENAI_MODEL for --ask (optional)")
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.ask:
        return _doctor_ask(args.ask, out_format=args.format, model=args.model)

    results = collect_diagnostics()
    if args.format == "json":
        import json

        print(json.dumps([r.__dict__ for r in results], indent=2))
    else:
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


# -------------------------- LLM-assisted help ---------------------------------
def _doctor_ask(question: str, *, out_format: str = "text", model: str | None = None) -> int:
    """Answer a user question using an LLM (optional).

    Requires the optional AI extra (openai) and OPENAI_API_KEY env var.
    """
    from pydantic import BaseModel, Field

    try:
        from .ai.llm import LLMError, generate_structured_openai
    except Exception as exc:  # pragma: no cover - optional import
        print(f"doctor: AI support not available: {exc}")
        return 2

    class DoctorAnswer(BaseModel):
        intent: str = Field(description="detected intent: diagnose|example|howto|other")
        summary: str
        steps: list[str] = Field(default_factory=list, description="concrete actions")
        commands: list[str] = Field(default_factory=list, description="shell commands to try")
        circuit_snippet: str | None = Field(
            default=None, description="optional SPICE netlist or Python snippet"
        )
        notes: list[str] = Field(default_factory=list)

    sys_prompt = (
        "You are Spicelab Doctor. You help with EDA tooling, ngspice/ltspice/xyce,"
        " Python errors, and small circuit examples. Be concise and give actionable steps."
    )
    user_prompt = (
        "User question:\n" + question + "\n\n"
        "If requesting an example, provide a minimal reproducible one."
    )

    try:
        res = generate_structured_openai(DoctorAnswer, sys_prompt, user_prompt, model=model)
    except LLMError as exc:  # pragma: no cover
        print(f"doctor: LLM error: {exc}")
        return 2

    ans = res.parsed
    if out_format == "json":
        import json

        if ans is not None:
            payload = {"model": res.model, **ans.model_dump()}
        else:
            payload = {"model": res.model, "raw": res.content}
        print(json.dumps(payload, indent=2))
        return 0

    if ans is None:
        print(res.content)
        return 0

    print("Doctor (" + res.model + "):")
    print("- " + ans.summary)
    if ans.steps:
        print("\nSteps:")
        for s in ans.steps:
            print(" • " + s)
    if ans.commands:
        print("\nCommands:")
        for c in ans.commands:
            print(" $ " + c)
    if ans.circuit_snippet:
        print("\nCircuit example:\n" + ans.circuit_snippet)
    if ans.notes:
        print("\nNotes:")
        for n in ans.notes:
            print(" - " + n)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
