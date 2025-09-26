from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

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


def _which_ltspice() -> str:
    env_exe = os.environ.get("SPICELAB_LTSPICE") or os.environ.get("LTSPICE_EXE")
    attempted: list[str] = []
    if env_exe:
        attempted.append(env_exe)
        if Path(env_exe).exists():
            return env_exe
    mac_paths = [
        "/Applications/LTspice.app/Contents/MacOS/LTspice",
        "/Applications/LTspice.app/Contents/MacOS/LTspiceXVII",
        "/Applications/LTspice.app/Contents/MacOS/LTspice64",
    ]
    for p in mac_paths:
        attempted.append(p)
        if Path(p).exists():
            return p
    exe = shutil.which("ltspice") or shutil.which("LTspice") or shutil.which("XVIIx64.exe")
    if exe:
        return exe
    hints = [
        "macOS (Homebrew cask): brew install --cask ltspice",
        "Windows: download from Analog Devices (LTspice) and add install dir to PATH",
        "Set SPICELAB_LTSPICE env var to the LTspice executable",
    ]
    raise EngineBinaryNotFound("ltspice", hints=hints, attempted=attempted)


def _write_deck(
    workdir: Path,
    title: str,
    netlist: str,
    directives: Sequence[str],
) -> tuple[Path, Path]:
    """Write a .cir deck with analysis directives appended."""

    deck = workdir / "deck.cir"
    log = workdir / "ltspice.log"
    body = _strip_final_end(netlist)
    with deck.open("w", encoding="utf-8") as f:
        f.write(f"* {title}\n")
        if body:
            f.write(body.rstrip() + "\n")
        for line in directives:
            f.write(line.rstrip() + "\n")
        f.write(".end\n")
    return deck, log


def run_directives(
    netlist: str,
    directives: Sequence[str],
    title: str = "spicelab_lt",
    *,
    keep_workdir: bool = True,
) -> RunResult:
    """Run LTspice headless on a generated deck.

    We request ASCII RAW with '-ascii'. Exact flags can vary per version/OS; if not
    supported, LTspice will still produce the default .raw which may be binary.
    """

    exe = _which_ltspice()
    workdir = Path(tempfile.mkdtemp(prefix="spicelab_lt_"))
    deck, log = _write_deck(workdir, title=title, netlist=netlist, directives=directives)

    # Expected RAW path: same base as deck, or as specified by LTspice default
    raw_out = workdir / "deck.raw"

    # Common CLI: `-b <deck>` runs in batch/headless mode. Newer builds accept
    # additional flags like `-Run`, but they must be ordered precisely and
    # differ between platforms. To stay compatible we only require `-b` and put
    # the deck path last, which works across macOS and Windows.
    cmd = [exe, "-b", str(deck)]
    proc = subprocess.run(cmd, cwd=str(workdir), capture_output=True, text=True)

    # Some LTspice builds write logs to stdout/stderr only
    with log.open("w", encoding="utf-8") as lf:
        if proc.stdout:
            lf.write(proc.stdout)
        if proc.stderr:
            lf.write("\n" + proc.stderr)

    raw_path: str | None = None
    if raw_out.exists():
        raw_path = str(raw_out)
    else:
        raws = sorted(workdir.glob("*.raw"))
        if raws:
            raw_path = str(raws[0])
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


class LtSpiceCLIAdapter(SimulatorAdapter):
    def __init__(self, *, keep_workdir: bool = True) -> None:
        self.keep_workdir = keep_workdir

    def run_directives(self, netlist_text: str, directives: Sequence[str]) -> RunResult:
        return run_directives(netlist_text, directives, keep_workdir=self.keep_workdir)


DEFAULT_ADAPTER = LtSpiceCLIAdapter()
