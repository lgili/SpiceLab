from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from ..core.types import AnalysisSpec, Probe
from ..engines.exceptions import EngineBinaryNotFound
from .base import RunArtifacts, RunResult, SimulatorAdapter


def _strip_final_end(netlist: str) -> str:
    lines = netlist.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[-1].strip().lower() == ".end":
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()
    return "\n".join(lines)


def _format_probe_signal(probe: Probe) -> str:
    target = probe.target.strip()
    if not target:
        return ""
    low = target.lower()
    if probe.kind == "voltage":
        if low.startswith("v("):
            return "V" + target[1:]
        return f"V({target})"
    if probe.kind == "current":
        if low.startswith("i("):
            return "I" + target[1:]
        return f"I({target})"
    return target


def _unique_preserve(seq: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _build_print_directives(
    analyses: Sequence[AnalysisSpec], probes: list[Probe] | None
) -> list[str]:
    if not analyses:
        return []

    requested: list[str] = []
    if probes:
        for probe in probes:
            sig = _format_probe_signal(probe)
            if sig:
                requested.append(sig)
    signals = _unique_preserve(requested)
    if not signals:
        signals = ["V(*)"]

    modes = {spec.mode for spec in analyses}
    prints: list[str] = []
    if any(mode == "tran" for mode in modes):
        tran_signals = _unique_preserve(signals)
        prints.append(".print tran format=csv " + " ".join(tran_signals))
    if any(mode == "ac" for mode in modes):
        ac_signals = _unique_preserve(signals)
        prints.append(".print ac format=csv " + " ".join(ac_signals))
    if any(mode == "dc" for mode in modes):
        prints.append(".print dc format=csv " + " ".join(signals))
    if any(mode == "op" for mode in modes):
        prints.append(".print op format=csv " + " ".join(signals))
    return prints


def _which_xyce() -> str:
    env_exe = os.environ.get("SPICELAB_XYCE") or os.environ.get("XYCE_EXE")
    attempted: list[str] = []
    if env_exe:
        attempted.append(env_exe)
        if Path(env_exe).exists():
            return env_exe
    exe = shutil.which("Xyce") or shutil.which("xyce")
    if exe:
        return exe
    hints = [
        "Download/build from https://xyce.sandia.gov/ (prebuilt packages for some distros)",
        "macOS (Homebrew, community tap): brew install xyce (if available)",
        "Set SPICELAB_XYCE env var (or XYCE_EXE) to the Xyce binary path",
    ]
    raise EngineBinaryNotFound("xyce", hints=hints, attempted=attempted)


def _write_deck(
    workdir: Path,
    title: str,
    netlist: str,
    directives: Sequence[str],
) -> Path:
    deck = workdir / "deck.cir"
    with deck.open("w", encoding="utf-8") as f:
        f.write(f"* {title}\n")
        body = _strip_final_end(netlist)
        if body:
            f.write(body.rstrip() + "\n")
        for line in directives:
            f.write(line.rstrip() + "\n")
        f.write(".end\n")
    return deck


def run_directives(
    netlist: str,
    directives: Sequence[str],
    title: str = "spicelab_xyce",
    *,
    keep_workdir: bool = True,
) -> RunResult:
    exe = _which_xyce()
    workdir = Path(tempfile.mkdtemp(prefix="spicelab_xy_"))
    deck = _write_deck(workdir, title=title, netlist=netlist, directives=directives)

    cmd = [exe, str(deck)]
    proc = subprocess.run(cmd, cwd=str(workdir), capture_output=True, text=True)

    # synthesize a log file from stdout/stderr
    log = workdir / "xyce.log"
    with log.open("w", encoding="utf-8") as lf:
        if proc.stdout:
            lf.write(proc.stdout)
        if proc.stderr:
            lf.write("\n" + proc.stderr)

    raw_path: str | None = None
    csv_candidates = sorted(workdir.glob(f"{deck.name}*.csv"))
    if csv_candidates:
        raw_path = str(csv_candidates[0])
    else:
        prn_candidates = sorted(workdir.glob(f"{deck.name}*.prn"))
        if prn_candidates:
            raw_path = str(prn_candidates[0])

    return RunResult(
        artifacts=RunArtifacts(
            netlist_path=str(deck),
            log_path=str(log),
            raw_path=raw_path,
            workdir=str(workdir) if keep_workdir else str(workdir),
        ),
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


class XyceCLIAdapter(SimulatorAdapter):
    def __init__(self, *, keep_workdir: bool = True) -> None:
        self.keep_workdir = keep_workdir

    def run_directives(self, netlist_text: str, directives: Sequence[str]) -> RunResult:
        return run_directives(netlist_text, directives, keep_workdir=self.keep_workdir)


DEFAULT_ADAPTER = XyceCLIAdapter()
