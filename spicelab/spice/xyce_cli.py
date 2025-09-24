from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from .base import RunArtifacts, RunResult, SimulatorAdapter


def _which_xyce() -> str:
    env_exe = os.environ.get("SPICELAB_XYCE")
    if env_exe and Path(env_exe).exists():
        return env_exe
    exe = shutil.which("Xyce") or shutil.which("xyce")
    if not exe:
        raise RuntimeError(
            "Xyce binary not found. Set SPICELAB_XYCE or install Xyce and add it to PATH."
        )
    return exe


def _write_deck(
    workdir: Path,
    title: str,
    netlist: str,
    directives: Sequence[str],
) -> Path:
    deck = workdir / "deck.cir"
    with deck.open("w", encoding="utf-8") as f:
        f.write(f"* {title}\n")
        f.write(netlist.rstrip() + "\n")
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

    # Xyce writes deck.cir.prn by default
    prn_out = workdir / (deck.name + ".prn")
    csv_out = workdir / (deck.name + ".csv")
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
    # Prefer CSV if produced, else PRN; we store the chosen one in raw_path field
    if csv_out.exists():
        raw_path = str(csv_out)
    elif prn_out.exists():
        raw_path = str(prn_out)

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
